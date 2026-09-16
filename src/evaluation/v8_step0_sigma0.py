"""
V8 Step 0 (pre-registered in PROTOCOL.md, 2026-09-15): is the M >= 12 instability a SIZE pathology
or a DISORDER pathology?

Result 30 measured frozen V7 leaving the physical set at M >= 12, but only at sigma = 0.15, because
no reference exists above M = 10 in the disordered model. Two explanations are confounded:

    (a) SIZE     -- the closure fails on scale alone, and should fail at sigma = 0 too.
    (b) DISORDER -- the failure needs broken permutation symmetry, and should vanish at sigma = 0.

This runs the frozen V7 at sigma = 0 across M, with no retraining and no re-tuning, and applies the
same reference-free bounds Result 30 used: per-emitter |<sz_i>| <= 1 and <a'a> >= 0.

THE CONFOUND, AND ITS CONTROL. sigma = 0 is itself outside the training distribution (V7 was trained
at sigma = 0.15), so a failure at sigma = 0 could be about the disorder width rather than the size.
The small-M rows are the control: if V7 is physical and accurate at sigma = 0 for M = 5, 8, 10 --
where exact references exist -- then sigma = 0 alone is not the problem, and any failure that appears
only as M grows is attributable to size.

Physicality is reference-free and cheap, so it runs first and answers the fork on its own. Accuracy
against the exact permutation-invariant reference is the separate, slower half.
"""
import numpy as np

from closure_v3 import ClosureApplyV3, ratio_transform
from closure_v6 import ClosureApplyV6, run_v6, lambdas
from local_run import observables
from pilot_eval import build_cell, m0_product, run_from, WC
from v7 import route, _load

GC = 0.7
REGIMES = [("POCKET", 0.03, 0.02), ("CONTROL", 0.15, 0.02)]
MS = [5, 8, 10, 12, 16, 20]
SIGMA0 = 0.0
SEED = 7          # irrelevant at sigma = 0 (every draw is w_i = wc) but kept for build_cell's signature


def per_site_z(eng, traj):
    cols = [eng.tpos[(0, 0, tuple(3 if s == i else 0 for s in range(eng.M)))] for i in range(eng.M)]
    return np.array([[row[c].real for c in cols] for row in traj])


def photon(eng, traj):
    return np.array([row[eng.tpos[(1, 1, tuple([0] * eng.M))]].real for row in traj])


def run_cell(M, ka, gp):
    eng, w, g = build_cell(M, GC, ka, gp, SIGMA0, SEED)
    det = w - WC                                  # identically zero at sigma = 0
    assert np.max(np.abs(det)) == 0.0, "sigma=0 must give zero detunings"
    regime = (GC, ka, gp)
    m0 = m0_product(eng, 1.0)
    _, t_plain = run_from(eng, m0, None)
    br = route(ka, gp)
    if br == "v6":
        _, t7, _ = run_v6(eng, m0, ClosureApplyV6(_load("v6"), eng, det, regime, xform=ratio_transform),
                          lambdas(ka, gp))
    else:
        _, t7 = run_from(eng, m0, ClosureApplyV3(_load("v3dens"), eng, det, regime, xform=ratio_transform))
    out = {"branch": br}
    for name, traj in (("plain", t_plain), ("V7", t7)):
        finite = np.isfinite(traj).all()
        if not finite:
            out[name] = dict(finite=False, z_min=np.nan, z_max=np.nan, n_min=np.nan,
                             n_peak=np.nan, physical=False, maxabs=np.inf)
            continue
        z, n = per_site_z(eng, traj), photon(eng, traj)
        out[name] = dict(
            finite=True,
            z_min=float(z.min()), z_max=float(z.max()), n_min=float(n.min()),
            n_peak=float(n.max()), maxabs=float(np.max(np.abs(traj))),
            physical=bool(z.min() >= -1.0 - 1e-9 and z.max() <= 1.0 + 1e-9 and n.min() >= -1e-9),
        )
    return out


def main():
    print("=" * 108)
    print("V8 STEP 0 -- frozen V7 at ZERO disorder: is the M>=12 instability about SIZE or about DISORDER?")
    print("=" * 108)
    print(f"  g_c={GC}, gamma_down=0.02, sigma=0 (identical emitters), fully inverted initial state")
    print("  bounds: -1 <= <sz_i> <= +1 per emitter, <a'a> >= 0   (reference-free, same as Result 30)\n")
    rows = {}
    for regime, ka, gp in REGIMES:
        print(f"--- {regime}  (kappa={ka}, gphi={gp})  V7 routes to {route(ka, gp).upper()} ---")
        print(f"{'M':>4} | {'plain phys':>10} {'plain z_min':>12} | {'V7 phys':>8} {'V7 z_min':>12} "
              f"{'V7 z_max':>12} {'V7 n_min':>11} {'V7 max|m|':>11}")
        for M in MS:
            r = run_cell(M, ka, gp)
            rows[(regime, M)] = r
            p, v = r["plain"], r["V7"]
            print(f"{M:>4} | {str(p['physical']):>10} {p['z_min']:>12.4f} | {str(v['physical']):>8} "
                  f"{v['z_min']:>12.4f} {v['z_max']:>12.4f} {v['n_min']:>11.3e} {v['maxabs']:>11.3e}",
                  flush=True)
        print()

    # verdict
    bad = [(rg, M) for (rg, M), r in rows.items() if not r["V7"]["physical"]]
    small = [(rg, M) for (rg, M) in bad if M <= 10]
    large = [(rg, M) for (rg, M) in bad if M >= 12]
    print("=" * 108)
    if not bad:
        print("OUTCOME O2 (DISORDER pathology): V7 stays physical at sigma = 0 at every M tested,")
        print("including sizes where it is measured to diverge at sigma = 0.15. The failure needs")
        print("broken permutation symmetry; it is not about scale alone.")
    elif small:
        print("CONFOUNDED: V7 already leaves the physical set at sigma = 0 for M <= 10, where it is")
        print("physical at sigma = 0.15. sigma = 0 is itself out of distribution, so this run cannot")
        print("separate size from disorder. Report as inconclusive; do not claim either outcome.")
    elif large:
        print("OUTCOME O1 (SIZE pathology): V7 is physical at sigma = 0 for M <= 10 (the control holds,")
        print("so sigma = 0 alone is not the cause) and leaves the physical set at large M anyway.")
        print("The instability is about scale. It is now reproducible against an exact reference at")
        print("large M, which is the precondition for diagnosing it.")
    print(f"  non-physical V7 cells: {sorted(bad) if bad else 'none'}")
    print("=" * 108)
    np.savez("results/v8_step0_sigma0.npz", rows=np.array([rows], dtype=object))
    print("saved results/v8_step0_sigma0.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
