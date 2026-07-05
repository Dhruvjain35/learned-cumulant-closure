"""
Size-transfer staircase (per review): M = 6,7,8,9,10 x seeds 7,8,9 at the V0 regime vs seeded mcsolve, to
locate WHERE stability begins degrading (campaign showed M=8 solid, M=10 marginal -- 1 of 2 seeds
unstable). Every row: plain, v2 (frozen), v3-RATIO with absolutes; mcsolve MC error stated. mcsolve refs
are cached by (M, seed), so already-computed references are reused, new ones generated here.
"""
import pickle
import time
import numpy as np

from scalable_engine import ScalableEngine
from closure_v3 import ClosureApplyV3, ratio_transform
from local_run import ClosureApply, observables, stable_ok
from pilot_eval import m0_product, run_from, nrmse, WC, NC
from campaign_eval import metrics, fmt, V0REG
from mc_ref import mc_reference
from exact_tc import draw_disorder

SEEDS = [7, 8, 9]
MS = [6, 7, 8, 9, 10]

if __name__ == "__main__":
    with open("results/lc_new.pkl", "rb") as fh:
        lc2 = pickle.load(fh)
    with open("results/lc_v3rat.pkl", "rb") as fh:
        lc3 = pickle.load(fh)
    gc, ka, gp, sg = V0REG
    print(f"staircase at V0 regime {V0REG}, M={MS}, seeds={SEEDS}, mcsolve ntraj=200 (cached by M,seed)")
    rows = {}
    for M in MS:
        for sd in SEEDS:
            t0 = time.time()
            w = draw_disorder(M, WC, sg, sd, "gauss")
            g = np.full(M, gc / np.sqrt(M))
            eng = ScalableEngine(M, NC, WC, w, g, kappa=ka, gdown=0.02, gphi=gp, Fsafe=6).compile()
            det = w - WC
            _, nref, zref, mcerr = mc_reference(M, NC, w, g, ntraj=200)
            m0 = m0_product(eng, 1.0)
            row = {}
            for name, c in {"plain": None, "v2": ClosureApply(lc2, eng, det),
                            "v3rat": ClosureApplyV3(lc3, eng, det, (gc, ka, gp),
                                                    xform=ratio_transform)}.items():
                _, traj = run_from(eng, m0, c)
                row[name] = metrics(eng, traj, nref, zref)
            rows[(M, sd)] = (row, mcerr)
            print(f"  M={M} seed {sd} (MCerr {mcerr:.3f}): plain[{fmt(row['plain'])}]  "
                  f"v2[{fmt(row['v2'])}]  v3rat[{fmt(row['v3rat'])}]  ({time.time()-t0:.0f}s)", flush=True)
    stab = {M: sum(rows[(M, s)][0]["v3rat"]["ok"] for s in SEEDS) for M in MS}
    print("\nv3-RATIO stability per M (of 3 seeds):", stab)
    np.savez_compressed("results/staircase.npz", rows=np.array(rows, dtype=object))
    print("saved results/staircase.npz")
