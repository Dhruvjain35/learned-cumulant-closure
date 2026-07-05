"""
FINAL VERDICT for V4 (density-frozen closure) on untouched seeds 11/12/13 (seed ledger, PROTOCOL.md):
these seeds had no part in diagnosis (7/8) or model selection (9/10). V4 = lc_v3dens.pkl exactly as
frozen by the 2x2; base V3 = lc_v3rat.pkl run alongside for the same-seed paired contrast.

Coverage: the full 5x6 pocket grid (stability + conditional NRMSE per seed, reported separately and
paired) + the 4 out-of-pocket sanity cells. Exact mesolve references. Nothing is tuned here; whatever
this says is the claim.
"""
import pickle
import time
import numpy as np

from closure_v3 import ClosureApplyV3, ratio_transform
from local_run import observables, stable_ok
from pilot_eval import build_cell, m0_product, exact_ref, nrmse, WC
from probe_pocket import run_instrumented, KAS, GPS
from fix2x2 import SANITY

M, GC, SG = 5, 0.7, 0.15
FINAL_SEEDS = [11, 12, 13]

if __name__ == "__main__":
    with open("results/lc_v3rat.pkl", "rb") as fh:
        v3 = pickle.load(fh)
    with open("results/lc_v3dens.pkl", "rb") as fh:
        v4 = pickle.load(fh)
    print(f"FINAL VERDICT: V3 vs V4(frozen density) on untouched seeds {FINAL_SEEDS}\n")
    rows = {}
    for ka in KAS:
        for gp in GPS:
            for sd in FINAL_SEEDS:
                eng, w, g = build_cell(M, GC, ka, gp, SG, sd)
                det = w - WC
                ne, ze, _ = exact_ref(M, w, g, ka, gp)
                for arm, lc in (("v3", v3), ("v4", v4)):
                    clo = ClosureApplyV3(lc, eng, det, (GC, ka, gp), xform=ratio_transform)
                    _, traj, _, _ = run_instrumented(eng, m0_product(eng, 1.0), clo)
                    ok = stable_ok(traj)
                    rows[(ka, gp, sd, arm)] = (ok, nrmse(observables(eng, traj)[0], ne) if ok else np.inf)
            line = " | ".join(
                f"{arm}:" + "/".join(("S" if rows[(ka, gp, s, arm)][0] else "X") for s in FINAL_SEEDS)
                for arm in ("v3", "v4"))
            print(f"  ka={ka:.3f} gp={gp:.3f}  {line}", flush=True)

    print("\nper-seed summary (stability | median conditional NRMSE):")
    for arm in ("v3", "v4"):
        for sd in FINAL_SEEDS:
            rs = [rows[(ka, gp, sd, arm)] for ka in KAS for gp in GPS]
            st = 100.0 * np.mean([r[0] for r in rs])
            nn = [r[1] for r in rs if r[0]]
            print(f"  {arm} seed {sd}: {st:5.1f}%  |  {np.median(nn):.3f}")
    for arm in ("v3", "v4"):
        rs = [rows[k] for k in rows if k[3] == arm]
        st = 100.0 * np.mean([r[0] for r in rs])
        nn = [r[1] for r in rs if r[0]]
        print(f"  {arm} POOLED: {st:5.1f}%  |  median {np.median(nn):.3f}")

    print("\nsanity cells (seed 11):")
    for (gc, ka, gp, sg) in SANITY:
        eng, w, g = build_cell(M, gc, ka, gp, sg, 11)
        det = w - WC
        ne, ze, _ = exact_ref(M, w, g, ka, gp)
        vals = []
        for arm, lc in (("v3", v3), ("v4", v4)):
            clo = ClosureApplyV3(lc, eng, det, (gc, ka, gp), xform=ratio_transform)
            _, traj, _, _ = run_instrumented(eng, m0_product(eng, 1.0), clo)
            vals.append(f"{arm}={nrmse(observables(eng, traj)[0], ne):.3f}"
                        if stable_ok(traj) else f"{arm}=UNSTABLE")
        print(f"  ({gc},{ka},{gp},{sg}): " + "  ".join(vals), flush=True)
    np.savez_compressed("results/final_verdict.npz", rows=np.array(rows, dtype=object))
    print("saved results/final_verdict.npz")
