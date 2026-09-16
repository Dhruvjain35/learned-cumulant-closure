"""
Separate the sigma = 0 confound from the size effect (the open caveat in Result 34).

Result 34's accuracy ladder runs at sigma = 0, because that is the only place the
permutation-invariant reference applies. But sigma = 0 is itself outside V7's training
distribution (trained at sigma = 0.15), so the error growth measured there could be about the
disorder width rather than about M. The M = 5 rows suggested the sigma = 0 penalty is small,
but one point is not a slope.

This measures the slope directly. Hold M FIXED at 5, where an exact mesolve reference exists for
any disorder, and sweep sigma from 0 to 0.20. If V7's burst-peak error is flat in sigma, then
sigma = 0 is benign and the growth seen in Result 34 is attributable to size.

Disorder draws use the burst-peak seeds 50/51/52 already used for the headline table, so nothing
new is consumed; sigma = 0 has no draw and is reported once.

Run:  python run.py src/evaluation/v8_sigma_sensitivity.py
"""
import numpy as np

from closure_v3 import ClosureApplyV3, ratio_transform
from closure_v6 import ClosureApplyV6, run_v6, lambdas
from exact_tc import draw_disorder
from pilot_eval import build_cell, m0_product, run_from, exact_ref, WC
from v7 import route, _load

GC, M = 0.7, 5
SIGMAS = [0.0, 0.05, 0.10, 0.15, 0.20]
SEEDS = [50, 51, 52]
REGIMES = [("POCKET", 0.03, 0.02), ("CONTROL", 0.15, 0.02)]


def peak_err(M, ka, gp, sg, seed):
    """Signed burst-peak relative error of V7 and of plain second order, vs exact mesolve."""
    eng, w, g = build_cell(M, GC, ka, gp, sg, seed)
    det, regime, m0 = w - WC, (GC, ka, gp), m0_product(eng, 1.0)
    n_ex = exact_ref(M, w, g, ka, gp)[0]
    ref = float(np.max(np.real(n_ex)))
    col = eng.tpos[(1, 1, tuple([0] * M))]
    out = {}
    for name in ("V7", "plain"):
        if name == "plain":
            _, tr = run_from(eng, m0, None)
        elif route(ka, gp) == "v6":
            _, tr, _ = run_v6(eng, m0, ClosureApplyV6(_load("v6"), eng, det, regime,
                                                      xform=ratio_transform), lambdas(ka, gp))
        else:
            _, tr = run_from(eng, m0, ClosureApplyV3(_load("v3dens"), eng, det, regime,
                                                     xform=ratio_transform))
        out[name] = (float(np.max([r[col].real for r in tr])) - ref) / ref if np.isfinite(tr).all() else np.nan
    return out, ref


def main():
    print("=" * 92)
    print(f"V8: sigma-sensitivity of the burst-peak error at FIXED M = {M} (exact mesolve reference)")
    print("=" * 92)
    print("  Question: is the error growth in Result 34 about size, or about sigma = 0 being")
    print("  out of distribution? If V7's error is flat in sigma, sigma = 0 is benign.\n")
    rows = {}
    for rg, ka, gp in REGIMES:
        print(f"--- {rg} (kappa={ka}, gphi={gp}), branch {route(ka, gp)} ---")
        print(f"{'sigma':>7} | {'V7 relerr (mean over seeds)':>28} {'plain relerr':>14}")
        for sg in SIGMAS:
            seeds = [0] if sg == 0.0 else SEEDS           # no draw at sigma = 0
            v7s, pls = [], []
            for sd in seeds:
                o, _ = peak_err(M, ka, gp, sg, sd)
                v7s.append(o["V7"]); pls.append(o["plain"])
            rows[(rg, sg)] = (float(np.mean(v7s)), float(np.mean(pls)))
            spread = f" (spread {np.max(v7s)-np.min(v7s):+.4f})" if len(v7s) > 1 else ""
            print(f"{sg:>7.2f} | {np.mean(v7s):>+28.4f}{spread} {np.mean(pls):>+14.4f}", flush=True)
        print()
    print("=" * 92)
    for rg, _, _ in REGIMES:
        e0 = abs(rows[(rg, 0.0)][0]); e15 = abs(rows[(rg, 0.15)][0])
        print(f"{rg}: |V7 err| at sigma=0 is {e0:.4f}; at sigma=0.15 it is {e15:.4f}  "
              f"-> sigma=0 penalty {e0-e15:+.4f}")
    print("\nCompare against the SIZE effect measured in Result 34 at sigma = 0:")
    print("  POCKET  |V7 err| 0.0083 (M=5) -> 0.0625 (M=12), a change of +0.054")
    print("  CONTROL |V7 err| 0.0028 (M=5) -> 0.0491 (M=12), a change of +0.046")
    np.savez("results/v8_sigma_sensitivity.npz", rows=np.array([rows], dtype=object))
    print("\nsaved results/v8_sigma_sensitivity.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
