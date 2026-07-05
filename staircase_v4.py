"""
Boundary-move proof, size axis: the M=6..10 staircase re-run with the FROZEN V4 (density) closure,
V3 alongside, same seeds (7/8/9 -- the original staircase comparison set), same cached seeded mcsolve
references. Question: did pocket-local training density also move the far-M stability boundary?
(Plausible mechanism: wider regime support -> wider dynamical-state support; but NOT assumed.)
"""
import pickle
import time
import numpy as np

from scalable_engine import ScalableEngine
from closure_v3 import ClosureApplyV3, ratio_transform
from local_run import stable_ok, observables
from pilot_eval import m0_product, run_from, nrmse, WC, NC
from campaign_eval import metrics, fmt, V0REG
from mc_ref import mc_reference
from exact_tc import draw_disorder

SEEDS = [7, 8, 9]          # override with --seeds 11,12,13 for the fresh-seed final-verdict run
MS = [6, 7, 8, 9, 10]

if __name__ == "__main__":
    import sys
    if "--seeds" in sys.argv:
        SEEDS = [int(s) for s in sys.argv[sys.argv.index("--seeds") + 1].split(",")]
    with open("results/lc_v3rat.pkl", "rb") as fh:
        v3 = pickle.load(fh)
    with open("results/lc_v3dens.pkl", "rb") as fh:
        v4 = pickle.load(fh)
    gc, ka, gp, sg = V0REG
    print(f"staircase re-run, V3 vs FROZEN V4, V0 regime, cached mcsolve refs, seeds {SEEDS}")
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
            for arm, lc in (("v3", v3), ("v4", v4)):
                _, traj = run_from(eng, m0, ClosureApplyV3(lc, eng, det, (gc, ka, gp),
                                                           xform=ratio_transform))
                row[arm] = metrics(eng, traj, nref, zref)
            rows[(M, sd)] = (row, mcerr)
            print(f"  M={M} seed {sd} (MCerr {mcerr:.3f}): v3[{fmt(row['v3'])}]  "
                  f"v4[{fmt(row['v4'])}]  ({time.time()-t0:.0f}s)", flush=True)
    for arm in ("v3", "v4"):
        stab = {M: sum(rows[(M, s)][0][arm]["ok"] for s in SEEDS) for M in MS}
        print(f"\n{arm} stability per M (of 3): {stab}")
    out = f"results/staircase_v4_s{'_'.join(map(str, SEEDS))}.npz" if SEEDS != [7, 8, 9] \
        else "results/staircase_v4.npz"
    np.savez_compressed(out, rows=np.array(rows, dtype=object))
    print(f"saved {out}")
