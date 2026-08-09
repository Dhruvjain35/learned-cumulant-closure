"""
Measure the positivity violation before attempting to fix it.

The manuscript reports that mean inversion undershoots -1 by 0.4-1.3% at M >= 12, and lists a
positivity constraint as future work. Cumulant truncations are not positivity-preserving in
general, so some violation is expected; what matters is how large it is, where in the trajectory
it happens, and whether it grows with system size.

These bounds need no exact reference, which is what makes this checkable at sizes where nothing
else is:

    -1 <= <sz_i> <= +1        each emitter's inversion lies between fully down and fully up
    <a^d a> >= 0              photon number cannot be negative

Any excursion outside them is unambiguously unphysical, and is the closure's fault rather than a
reference's noise. This script quantifies both for the frozen V7 and, as a control, for the plain
second-order closure -- because if the baseline violates them by more, a violation in V7 is a
shared property of cumulant truncation rather than a defect introduced by learning.

Run:  python src/analysis/positivity_audit.py [--smoke]
"""
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("engines", "closures", "evaluation", "analysis", "training"):
    sys.path.insert(0, os.path.join(_HERE, "..", sub))

from pilot_eval import m0_product, run_from, WC, NC, GDOWN  # noqa: E402
from scalable_engine import ScalableEngine  # noqa: E402
from exact_tc import draw_disorder  # noqa: E402
from closure_v3 import ClosureApplyV3, ratio_transform  # noqa: E402
from closure_v6 import ClosureApplyV6, run_v6, lambdas  # noqa: E402
from v7 import route, _load  # noqa: E402

GC, KAPPA, GPHI, SIGMA, SEED = 0.7, 0.03, 0.02, 0.15, 7


def cell(M, seed=SEED):
    w = draw_disorder(M, WC, SIGMA, seed, "gauss")
    g = np.full(M, GC / np.sqrt(M))
    eng = ScalableEngine(M, NC, WC, w, g, kappa=KAPPA, gdown=GDOWN, gphi=GPHI, Fsafe=6).compile()
    return eng, w, g


def per_site_z(eng, traj):
    """All <sz_i>(t), not just the mean -- the bound applies to each emitter separately, and an
    average can look legal while individual sites do not."""
    M = eng.M
    cols = [eng.tpos[(0, 0, tuple(3 if s == i else 0 for s in range(M)))] for i in range(M)]
    return np.array([[row[c].real for c in cols] for row in traj])


def photon(eng, traj):
    c = eng.tpos[(1, 1, tuple([0] * eng.M))]
    return np.array([row[c].real for row in traj])


def audit(M, seed=SEED):
    eng, w, g = cell(M, seed)
    det = w - WC
    regime = (GC, KAPPA, GPHI)
    out = {}

    _, tp = run_from(eng, m0_product(eng, 1.0), None)
    br = route(KAPPA, GPHI)
    if br == "v6":
        _, t7, _ = run_v6(eng, m0_product(eng, 1.0),
                          ClosureApplyV6(_load("v6"), eng, det, regime, xform=ratio_transform),
                          lambdas(KAPPA, GPHI))
    else:
        _, t7 = run_from(eng, m0_product(eng, 1.0),
                         ClosureApplyV3(_load("v3dens"), eng, det, regime, xform=ratio_transform))

    for name, traj in (("plain", tp), ("V7", t7)):
        z = per_site_z(eng, traj)
        n = photon(eng, traj)
        out[name] = dict(
            z_min=float(z.min()), z_max=float(z.max()),
            under=float(max(0.0, -1.0 - z.min())),      # how far below -1
            over=float(max(0.0, z.max() - 1.0)),        # how far above +1
            n_min=float(n.min()),
            frac_bad=float(np.mean((z < -1.0) | (z > 1.0))),
        )
    out["branch"] = br
    return out


def main():
    smoke = "--smoke" in sys.argv
    Ms = [5, 8] if smoke else [5, 8, 10, 12, 16, 20]
    seeds = [SEED] if smoke else [7, 8, 9]

    print("Positivity audit -- bounds that need no exact reference")
    print(f"  -1 <= <sz_i> <= 1 per emitter, <a^d a> >= 0;  g_c={GC} kappa={KAPPA} gphi={GPHI}")
    print("  Swept over seeds: the documented stability boundary in this corner is seed-dependent,")
    print("  so a single seed would misrepresent it in either direction.")
    print()
    print(f"  {'M':>3} {'closure':>7} {'seeds OK':>9} {'worst min <sz>':>15} "
          f"{'worst below -1':>15} {'worst min n':>12}")
    rows = []
    for M in Ms:
        per = {}
        for s in seeds:
            a = audit(M, seed=s)
            for name in ("plain", "V7"):
                rows.append(dict(M=M, closure=name, seed=s, **a[name]))
                per.setdefault(name, []).append(a[name])
        for name in ("plain", "V7"):
            ds = per[name]
            nok = sum(1 for d in ds if d["under"] <= 1e-9 and d["over"] <= 1e-9
                      and d["n_min"] >= -1e-9)
            worst = max(ds, key=lambda d: d["under"] + d["over"])
            flag = "  <-- violates" if nok < len(ds) else ""
            print(f"  {M:>3} {name:>7} {nok:>5}/{len(ds):<3} {worst['z_min']:>15.4f} "
                  f"{worst['under']:>15.3e} {worst['n_min']:>12.3e}{flag}")

    print()
    v7 = [r for r in rows if r["closure"] == "V7"]
    pl = [r for r in rows if r["closure"] == "plain"]
    print(f"  V7    : {sum(1 for r in v7 if r['under'] <= 1e-9 and r['n_min'] >= -1e-9)}"
          f"/{len(v7)} (M, seed) cells physical")
    print(f"  plain : {sum(1 for r in pl if r['under'] <= 1e-9 and r['n_min'] >= -1e-9)}"
          f"/{len(pl)} cells physical")
    bad = sorted({(r["M"], r["seed"]) for r in v7 if r["under"] > 1e-9 or r["n_min"] < -1e-9})
    print(f"  V7 failures at (M, seed): {bad if bad else 'none'}")

    os.makedirs("results", exist_ok=True)
    np.savez_compressed("results/positivity_audit.npz", rows=np.array(rows, dtype=object))
    print("  saved results/positivity_audit.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
