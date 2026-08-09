"""
How long does "stable" last?

Every result in this repository is measured over t <= 40, which earlier notes describe as
about 1.2 cavity lifetimes at the slowest cell, with the caveat that longer horizons are
uncharacterised. That caveat matters more than it looks: a closure can be perfectly well behaved
across the burst and then drift, oscillate, or diverge later, and a reader is entitled to ask
whether t <= 40 was chosen because it is physically sufficient or because it is where things
still work.

This script extends the horizon to t = 80 and t = 160 and asks two questions at each:

    Does the rollout stay finite and bounded?
    Does it stay PHYSICAL -- every <sz_i> within [-1, 1], photon number non-negative?

The second question needs no exact reference, which is the point: it remains checkable at
horizons and sizes where nothing can be compared against. Where an exact reference IS affordable
(small M) we also report the trajectory error, so drift can be distinguished from mere
instability.

TMAX and NT are module constants in pilot_eval, so they are patched here before the engine and
closures read them, and restored afterwards.

Run:  python src/evaluation/horizon_tests.py [--smoke]
"""
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("engines", "closures", "evaluation", "analysis", "training"):
    sys.path.insert(0, os.path.join(_HERE, "..", sub))

import qutip as qt  # noqa: E402
import pilot_eval  # noqa: E402
import local_run  # noqa: E402
from exact_tc import build_ops, build_H, collapse_ops, draw_disorder  # noqa: E402
from scalable_engine import ScalableEngine  # noqa: E402
from closure_v3 import ClosureApplyV3, ratio_transform  # noqa: E402
from closure_v6 import ClosureApplyV6, run_v6, lambdas  # noqa: E402
from v7 import route, _load  # noqa: E402

GC, KAPPA, GPHI, SIGMA, SEED = 0.7, 0.03, 0.02, 0.15, 7
WC, GDOWN, NC = pilot_eval.WC, pilot_eval.GDOWN, pilot_eval.NC


class Horizon:
    """Temporarily extend the integration window that pilot_eval / local_run read."""

    def __init__(self, tmax, nt):
        self.tmax, self.nt = tmax, nt

    def __enter__(self):
        self.old = (pilot_eval.TMAX, pilot_eval.NT, local_run.NT)
        pilot_eval.TMAX, pilot_eval.NT, local_run.NT = self.tmax, self.nt, self.nt
        return self

    def __exit__(self, *a):
        pilot_eval.TMAX, pilot_eval.NT, local_run.NT = self.old
        return False


def run_both(M, tmax, nt, seed=SEED):
    """Plain second order and frozen V7 over the requested horizon."""
    w = draw_disorder(M, WC, SIGMA, seed, "gauss")
    g = np.full(M, GC / np.sqrt(M))
    eng = ScalableEngine(M, NC, WC, w, g, kappa=KAPPA, gdown=GDOWN, gphi=GPHI, Fsafe=6).compile()
    det, regime = w - WC, (GC, KAPPA, GPHI)
    m0 = pilot_eval.m0_product(eng, 1.0)

    _, tp = pilot_eval.run_from(eng, m0, None)
    if route(KAPPA, GPHI) == "v6":
        # run_v6 takes its own tmax/nt with defaults of 40.0/200, so patching the module
        # constants does NOT reach it. They must be passed explicitly, or the V6 branch is
        # silently evaluated over the standard window while the plain baseline is extended --
        # which is exactly the bug this line fixes.
        _, t7, _ = run_v6(eng, m0, ClosureApplyV6(_load("v6"), eng, det, regime,
                                                  xform=ratio_transform), lambdas(KAPPA, GPHI),
                          tmax=tmax, nt=nt)
    else:
        _, t7 = pilot_eval.run_from(eng, m0, ClosureApplyV3(_load("v3dens"), eng, det, regime,
                                                            xform=ratio_transform))
    return eng, w, g, tp, t7


def physical(eng, traj):
    """Bounds that hold with no reference: |<sz_i>| <= 1 and <a^d a> >= 0."""
    M = eng.M
    zc = [eng.tpos[(0, 0, tuple(3 if s == i else 0 for s in range(M)))] for i in range(M)]
    nc = eng.tpos[(1, 1, tuple([0] * M))]
    z = np.array([[row[c].real for c in zc] for row in traj])
    n = np.array([row[nc].real for row in traj])
    finite = bool(np.isfinite(traj).all())
    return dict(finite=finite,
                bounded=bool(finite and np.max(np.abs(traj)) < 1e2),
                z_under=float(max(0.0, -1.0 - z.min())),
                z_over=float(max(0.0, z.max() - 1.0)),
                n_min=float(n.min()))


def exact_n(M, w, g, tmax, nt):
    ops = build_ops(M, NC)
    H = build_H(ops, WC, w, g)
    c = collapse_ops(ops, kappa=KAPPA, gdown=GDOWN, gphi=GPHI)
    psi0 = qt.tensor(qt.basis(NC, 0), *[qt.basis(2, 0) for _ in range(M)])
    ts = np.linspace(0, tmax, nt)
    e = qt.mesolve(H, psi0, ts, c_ops=c, e_ops=[ops["a"].dag() * ops["a"]],
                   options={"progress_bar": False}).expect
    return np.array(e[0]).real


def main():
    smoke = "--smoke" in sys.argv
    Ms = [3] if smoke else [3, 5, 8]
    horizons = [(40.0, 200), (80.0, 400)] if smoke else [(40.0, 200), (80.0, 400), (160.0, 800)]

    print("Horizon tests -- does the rollout stay finite AND physical past t = 40?")
    print(f"  g_c={GC} kappa={KAPPA} gphi={GPHI} sigma={SIGMA} seed={SEED}")
    print()
    print(f"  {'M':>3} {'tmax':>6} {'closure':>7} {'finite':>7} {'bounded':>8} "
          f"{'z below -1':>11} {'min n':>10} {'nrmse vs exact':>15}")

    rows = []
    for M in Ms:
        for (tmax, nt) in horizons:
            with Horizon(tmax, nt):
                eng, w, g, tp, t7 = run_both(M, tmax, nt)
                ne = exact_n(M, w, g, tmax, nt) if M <= 5 else None
                for name, traj in (("plain", tp), ("V7", t7)):
                    ph = physical(eng, traj)
                    nn = np.array([row[eng.tpos[(1, 1, tuple([0] * M))]].real for row in traj])
                    err = (float(np.sqrt(np.mean((nn - ne) ** 2)) /
                                 max(np.sqrt(np.mean(ne ** 2)), 1e-12))
                           if ne is not None and len(nn) == len(ne) else float("nan"))
                    rows.append(dict(M=M, tmax=tmax, closure=name, nrmse=err, **ph))
                    bad = (not ph["bounded"]) or ph["z_under"] > 1e-6 or ph["n_min"] < -1e-6
                    print(f"  {M:>3} {tmax:>6.0f} {name:>7} {str(ph['finite']):>7} "
                          f"{str(ph['bounded']):>8} {ph['z_under']:>11.2e} {ph['n_min']:>10.2e} "
                          f"{err:>15.4f}{'  <-- unphysical' if bad else ''}")

    print()
    v7 = [r for r in rows if r["closure"] == "V7"]
    for tmax in sorted({r["tmax"] for r in v7}):
        sel = [r for r in v7 if r["tmax"] == tmax]
        print(f"  t<={tmax:>5.0f}: V7 bounded in {sum(r['bounded'] for r in sel)}/{len(sel)} cells, "
              f"worst z undershoot {max(r['z_under'] for r in sel):.2e}")

    os.makedirs("results", exist_ok=True)
    np.savez_compressed("results/horizon_tests.npz", rows=np.array(rows, dtype=object))
    print("  saved results/horizon_tests.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
