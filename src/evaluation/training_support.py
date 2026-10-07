"""
Training support of the frozen V4/V6 closures, and a geometric classifier for evaluation points.

Pure geometry: no simulation, no model, no outcome data. Written for the held-out-parameter evaluation
(PROTOCOL.md, "Publication finalization P3") so that "held out" is decided by code, not by prose.

SOURCE OF TRUTH. The grid is imported from the two generators that produced the training data, not
re-typed here:
  gen_data_v3.py        main grid  g_c x kappa x gamma_phi x sigma, M = 2..5      (54 tuples)
  gen_pocket_density.py pocket     g_c = 0.7 only, kappa x gamma_phi x sigma      (27 tuples)
V4 (lc_v3dens.pkl, fitted in src/analysis/fix2x2.py) was trained on main + pocket. V6 (lc_v6.pkl,
src/training/fit_v56.py) on an area-balanced subsample of the same union (train_v6.npz: 972 configs, the
same 81 (g_c, kappa, gamma_phi, sigma) tuples at M = 2..5, no Fock skips). All training used M = 2..5,
gamma_down = 0.02, Gaussian disorder, the fully inverted initial state, and training disorder seeds {0,1,2}
(rng seeds sd*100 + round(100 sigma)).

CLASSES (for a point p = (g_c, kappa, gamma_phi, sigma) at fixed M; computed in raw parameter units):
  A  on the training grid: p is one of the 81 training tuples.
  B  interpolation: not on the grid, inside the convex hull of the 81 tuples.
  C  one-axis extrapolation: outside the hull, but moving ONE coordinate to some value inside that axis's
     training range puts it inside the hull.
  D  multi-axis: outside the hull and no single-coordinate move fixes it.
Caveat recorded with the classes: the closure's features use ratios (Delta/kappa, g_c/kappa, gamma_phi/kappa)
with kappa raw, so "inside the hull in raw parameters" is not the same as "inside the training distribution of
the features". The classes are a statement about physical parameters only.
"""
import itertools
import numpy as np
from scipy.optimize import linprog

from gen_data_v3 import GCOLLS, KAPPAS, GPHIS, SIGMAS
import gen_pocket_density as _pocket

AXES = ("g_c", "kappa", "gamma_phi", "sigma")


def training_tuples():
    main = set(itertools.product(GCOLLS, KAPPAS, GPHIS, SIGMAS))
    pocket = set(itertools.product([_pocket.GC], _pocket.KAS, _pocket.GPS, _pocket.SIGMAS))
    return sorted({tuple(float(x) for x in t) for t in main | pocket})


TRAIN = np.array(training_tuples())
AXIS_VALUES = [sorted(set(TRAIN[:, i])) for i in range(4)]
AXIS_RANGE = [(min(v), max(v)) for v in AXIS_VALUES]


def in_hull(p, pts=TRAIN, tol=1e-9):
    """Is p a convex combination of the rows of pts? (LP feasibility.)"""
    n = len(pts)
    A_eq = np.vstack([pts.T, np.ones(n)])
    b_eq = np.concatenate([np.asarray(p, float), [1.0]])
    res = linprog(np.zeros(n), A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * n, method="highs")
    return bool(res.status == 0 and np.allclose(A_eq @ res.x, b_eq, atol=1e-7))


def on_grid(p):
    return any(np.allclose(p, t, atol=1e-12) for t in TRAIN)


def classify(p, nscan=401):
    """Return a dict: class letter + per-axis facts. Deterministic."""
    p = tuple(float(x) for x in p)
    off_grid_axes = [AXES[i] for i in range(4) if not any(np.isclose(p[i], v) for v in AXIS_VALUES[i])]
    out_of_range_axes = [AXES[i] for i in range(4)
                         if not (AXIS_RANGE[i][0] - 1e-12 <= p[i] <= AXIS_RANGE[i][1] + 1e-12)]
    grid = on_grid(p)
    hull = in_hull(p)
    one_axis_fix = []
    if not hull:
        for i in range(4):
            lo, hi = AXIS_RANGE[i]
            for v in np.linspace(lo, hi, nscan):
                q = list(p); q[i] = v
                if in_hull(q):
                    one_axis_fix.append(AXES[i]); break
    if grid:
        cls = "A"
    elif hull:
        cls = "B"
    elif one_axis_fix:
        cls = "C"
    else:
        cls = "D"
    return dict(cls=cls, on_grid=grid, in_hull=hull, off_grid_axes=off_grid_axes,
                out_of_range_axes=out_of_range_axes, one_axis_fixes=one_axis_fix)


def self_test():
    """Known facts, asserted: the headline cells and the support size."""
    assert len(TRAIN) == 81, len(TRAIN)
    assert classify((0.70, 0.15, 0.02, 0.15))["cls"] == "A"          # CONTROL is a training-grid point
    c = classify((0.70, 0.03, 0.02, 0.15))                            # POCKET: off-grid tuple, outside hull
    assert c["cls"] == "C" and not c["out_of_range_axes"], c
    assert classify((0.70, 0.04, 0.05, 0.15))["cls"] == "A"           # a pocket-grid point
    assert classify((0.55, 0.10, 0.06, 0.10))["cls"] == "B"
    return True


if __name__ == "__main__":
    self_test()
    print(f"training support: {len(TRAIN)} tuples; ranges "
          + ", ".join(f"{AXES[i]} [{AXIS_RANGE[i][0]}, {AXIS_RANGE[i][1]}]" for i in range(4)))
    print("self-test PASS")
