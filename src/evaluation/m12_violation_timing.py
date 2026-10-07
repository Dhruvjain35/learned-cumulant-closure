"""
EXPLORATORY (not pre-registered; noted as optional in PROTOCOL P3): when does frozen V7 first leave the physical
set in the disordered M = 12 cells of Result 37, and is that after its burst peak?

Closure-only; no reference solve. Rebuilds exactly the Result 37 closure runs (build_cell(12, 0.7, kappa, 0.02,
0.15, seed 50), the same branch calls as v8_disordered_m12.closure_peak, t <= 40, 200 points) and reports:
  - V7 burst peak n_peak and its time, both as Result 37 computed it (max over the whole run) and within t <= 10;
  - the first time any per-emitter |<sz_i>| > 1 + 1e-6 or <a'a> < -1e-6;
  - whether the burst peak precedes that first violation.
Result 37's V7 relative errors (+0.0573 POCKET, +0.0400 CONTROL against the 400-trajectory references
8.1063 / 6.8082) are recomputed as a regression check.

    python run.py src/evaluation/m12_violation_timing.py
"""
import numpy as np

from closure_v3 import ClosureApplyV3, ratio_transform
from closure_v6 import ClosureApplyV6, run_v6, lambdas
from pilot_eval import build_cell, m0_product, run_from, WC
from v7 import route, _load

M, GC, SIGMA, SEED, TOL = 12, 0.7, 0.15, 50, 1e-6
REF = {"POCKET": (0.03, 0.02, 8.1063, 0.0573), "CONTROL": (0.15, 0.02, 6.8082, 0.0400)}


def main():
    L = []
    say = lambda s="": (print(s, flush=True), L.append(s))
    say("M = 12 DISORDERED V7 VIOLATION TIMING (exploratory; Result 37 cells, seed 50, sigma 0.15, t <= 40)")
    for rg, (ka, gp, nref, err37) in REF.items():
        eng, w, g = build_cell(M, GC, ka, gp, SIGMA, SEED)
        det, regime, m0 = w - WC, (GC, ka, gp), m0_product(eng, 1.0)
        br = route(ka, gp)
        if br == "v6":
            ts, tr, _ = run_v6(eng, m0, ClosureApplyV6(_load("v6"), eng, det, regime, xform=ratio_transform),
                               lambdas(ka, gp))
        else:
            ts, tr = run_from(eng, m0, ClosureApplyV3(_load("v3dens"), eng, det, regime, xform=ratio_transform))
        tp = eng.tpos
        n = np.array([r[tp[(1, 1, tuple([0] * M))]].real for r in tr])
        zc = [tp[(0, 0, tuple(3 if s == i else 0 for s in range(M)))] for i in range(M)]
        sz = np.array([[r[c].real for c in zc] for r in tr])
        bad = (np.max(np.abs(sz), axis=1) > 1 + TOL) | (n < -TOL) | ~np.isfinite(n)
        t_viol = float(ts[np.argmax(bad)]) if bad.any() else np.nan
        i_all = int(np.argmax(n))
        w10 = ts <= 10.0 + 1e-12
        i10 = int(np.argmax(np.where(w10, n, -np.inf)))
        err_all = (n[i_all] - nref) / nref
        say(f"\n{rg} (kappa {ka}, gphi {gp}, branch {br}), {len(ts)} points to t = {ts[-1]:.0f}")
        say(f"  burst peak (max over t <= 10):   n = {n[i10]:.4f} at t = {ts[i10]:.2f}")
        say(f"  max over whole run (as R37):     n = {n[i_all]:.4f} at t = {ts[i_all]:.2f}  -> relerr {err_all:+.4f} "
            f"(Result 37: {err37:+.4f}; reproduced: {abs(err_all - err37) < 6e-5})")
        say(f"  first physicality violation:     " + (f"t = {t_viol:.2f}" if np.isfinite(t_viol) else "none to t = 40"))
        say(f"  max per-site |<sz_i>| over run:  {np.max(np.abs(sz)):.3e}; min n {np.min(n):.3e}")
        if np.isfinite(t_viol):
            say(f"  burst precedes first violation:  {ts[i10] < t_viol}  (margin {t_viol - ts[i10]:.2f})")
    with open("results/m12_violation_timing.log", "w") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
