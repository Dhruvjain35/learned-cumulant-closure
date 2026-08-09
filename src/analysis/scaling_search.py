"""
Can a change of variables make the closure transfer to large M?

BACKGROUND, INCLUDING A MISTAKE WORTH RECORDING
-----------------------------------------------
transfer_collapse.py compared kappa3 across M in two coordinate systems and reported that the
estimator's own features collapse (1.12x) while the intensive context does not (3.07x). The
second half of that comparison was not a fair test: it binned by INTENSIVE features while
keeping an EXTENSIVE target. An extensive quantity cannot be a function of intensive ones, so
that arm was guaranteed to fail regardless of the physics. The conclusion drawn from it -- that
the intensive route is a dead end -- was therefore unsupported.

This script runs the test properly: it rescales the target and the features TOGETHER and asks,
for each candidate scaling, two separate questions.

    COLLAPSE  -- is the map the same function at every M?
                 (across-M spread of bin means, over within-M scatter; ~1 is good)
    OVERLAP   -- do large-M samples land inside the range small-M training covered?
                 (fraction of M=5 samples inside the M=2,3 feature range; ~1 is good)

Both matter, and they are different. The estimator's raw features already COLLAPSE well, but they
do not OVERLAP: <a^d a> grows with M, so a big system leaves the window a small system defined,
and the model extrapolates exactly where the burst peak lives. A scaling that achieves both
turns size-extrapolation into interpolation, which is the only route to large M that does not
require training data we cannot generate.

Nothing here is trained. This is a property of the exact data, so a negative result is just as
informative: if no scaling gives both, the ceiling is physical and should be reported as such
rather than engineered around.

Run:  python src/analysis/scaling_search.py [--smoke]
"""
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from kappa3_trace import solve_cell, kappa3_photon_inversion, NCS, WC  # noqa: E402


def raw_samples(M, tmax, nt):
    """Per-(time, emitter) exact quantities, before any scaling choice is made."""
    S = solve_cell(M, NCS[M], tmax=tmax, nt=nt)
    k3 = np.real(kappa3_photon_inversion(S))          # (NT, M)
    ones = np.ones((1, M))
    return dict(
        M=M,
        n=(S["n"][:, None] * ones).ravel(),           # <a^d a>, extensive
        zi=S["sz"].ravel(),                           # <sz_i>, intensive
        zbar=(S["sz"].mean(axis=1)[:, None] * ones).ravel(),
        det=((S["w"] - WC)[None, :] * np.ones((S["n"].size, 1))).ravel(),
        k3=k3.ravel(),
    )


# Each scaling returns (x, y, target). M is passed so the scaling may use it.
SCALINGS = {
    "raw:  k3  vs (n, z)":            lambda d: (d["n"], d["zi"], d["k3"]),
    "per-emitter:  k3/M vs (n/M, z)": lambda d: (d["n"] / d["M"], d["zi"], d["k3"] / d["M"]),
    "sqrt:  k3/sqrtM vs (n/sqrtM,z)": lambda d: (d["n"] / np.sqrt(d["M"]), d["zi"],
                                                 d["k3"] / np.sqrt(d["M"])),
    "photon-normalised: k3/n vs(n/M,z)": lambda d: (d["n"] / d["M"], d["zi"],
                                                    d["k3"] / np.maximum(d["n"], 1e-9)),
    "k3/M vs (n/M, zbar)":            lambda d: (d["n"] / d["M"], d["zbar"], d["k3"] / d["M"]),
}


def collapse(x, y, t, Ms, nb=14, minn=12):
    """Across-M spread of bin means, divided by mean within-M scatter."""
    xb = np.linspace(x.min(), x.max(), nb + 1)
    yb = np.linspace(y.min(), y.max(), nb + 1)
    ix = np.clip(np.digitize(x, xb) - 1, 0, nb - 1)
    iy = np.clip(np.digitize(y, yb) - 1, 0, nb - 1)
    across, within, nbins = [], [], 0
    for a in range(nb):
        for b in range(nb):
            m = (ix == a) & (iy == b)
            if m.sum() < minn:
                continue
            present = np.unique(Ms[m])
            if len(present) < 2:
                continue
            across.append(np.std([t[m & (Ms == q)].mean() for q in present]))
            within.append(np.mean([t[m & (Ms == q)].std() for q in present]))
            nbins += 1
    if not across:
        return np.nan, 0
    return float(np.mean(across) / max(np.mean(within), 1e-12)), nbins


def overlap(x, y, Ms, train, test, peak_only=False, xall=None):
    """Fraction of `test` samples inside the axis-aligned feature box spanned by `train`."""
    tr = np.isin(Ms, train)
    te = Ms == test
    if peak_only:
        thr = np.quantile(x[te], 0.9)
        te = te & (x >= thr)
    inside = ((x[te] >= x[tr].min()) & (x[te] <= x[tr].max()) &
              (y[te] >= y[tr].min()) & (y[te] <= y[tr].max()))
    return float(inside.mean())


def main():
    smoke = "--smoke" in sys.argv
    Ms = [2, 3, 4] if smoke else [2, 3, 4, 5]
    tmax, nt = (20.0, 201) if smoke else (40.0, 401)

    data = [raw_samples(M, tmax, nt) for M in Ms]
    merged = {k: np.concatenate([d[k] for d in data]) for k in ("n", "zi", "zbar", "det", "k3")}
    merged["M"] = np.concatenate([np.full(d["k3"].size, d["M"]) for d in data])

    print("Scaling search -- both properties must hold for large-M transfer")
    print(f"  sizes {Ms}, exact solves, {merged['k3'].size} samples total")
    print()
    print(f"  {'scaling':<34} {'collapse':>9} {'overlap':>8} {'peak-only':>10}")
    print(f"  {'':<34} {'(~1 good)':>9} {'(~1 good)':>8} {'overlap':>10}")

    rows = []
    for name, fn in SCALINGS.items():
        x, y, t = fn(merged)
        keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(t)
        x, y, t, MM = x[keep], y[keep], t[keep], merged["M"][keep]
        c, nb = collapse(x, y, t, MM)
        ov = overlap(x, y, MM, train=[2, 3], test=max(Ms))
        ovp = overlap(x, y, MM, train=[2, 3], test=max(Ms), peak_only=True)
        rows.append((name, c, ov, ovp, nb))
        print(f"  {name:<34} {c:>9.3f} {100 * ov:>7.1f}% {100 * ovp:>9.1f}%")

    print()
    good = [r for r in rows if np.isfinite(r[1]) and r[1] < 1.6 and r[3] > 0.85]
    if good:
        best = min(good, key=lambda r: r[1])
        print(f"  BEST: {best[0]}  (collapse {best[1]:.3f}, burst-peak overlap {100*best[3]:.1f}%)")
        print("  -> in these variables a small-M fit INTERPOLATES at larger M, including at the peak.")
    else:
        print("  No scaling achieves both. The size ceiling is then a property of the physics,")
        print("  not of the parameterisation, and should be reported rather than engineered around.")

    os.makedirs("results", exist_ok=True)
    np.savez_compressed("results/scaling_search.npz",
                        names=np.array([r[0] for r in rows]),
                        collapse=np.array([r[1] for r in rows]),
                        overlap=np.array([r[2] for r in rows]),
                        overlap_peak=np.array([r[3] for r in rows]))
    print("  saved results/scaling_search.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
