"""Plain 2nd-order cumulant (closure = None) at the final ladder points/seeds (32/33/34) -- the one rung
the six-model eval didn't carry, needed so the locked ladder figure is one consistent seed set."""
import numpy as np
from local_run import observables, stable_ok
from pilot_eval import build_cell, m0_product, run_from, exact_ref, nrmse, WC
from baselines_eval import POINTS, SANITY

SEEDS = [32, 33, 34]
GC, SG, M5 = 0.7, 0.15, 5

if __name__ == "__main__":
    res = {}
    for (rtag, ka, gp) in POINTS:
        for sd in SEEDS:
            eng, w, g = build_cell(M5, GC, ka, gp, SG, sd)
            ne, ze, _ = exact_ref(M5, w, g, ka, gp)
            _, traj = run_from(eng, m0_product(eng, 1.0), None)
            ok = stable_ok(traj)
            v = nrmse(observables(eng, traj)[0], ne) if ok else np.inf
            res[(rtag, ka, gp, sd)] = (ok, v)
            print(f"[{rtag}] ka={ka:.3f} gp={gp:.3f} s{sd}: plain={'X' if not ok else f'{v:.3f}'}",
                  flush=True)
    for (gc, ka, gp, sg) in SANITY:
        eng, w, g = build_cell(M5, gc, ka, gp, sg, SEEDS[0])
        ne, ze, _ = exact_ref(M5, w, g, ka, gp)
        _, traj = run_from(eng, m0_product(eng, 1.0), None)
        ok = stable_ok(traj)
        v = nrmse(observables(eng, traj)[0], ne) if ok else np.inf
        res[("sanity", gc, ka, gp, sg)] = (ok, v)
        print(f"[sanity] ({gc},{ka},{gp},{sg}): plain={'X' if not ok else f'{v:.3f}'}", flush=True)
    np.savez_compressed("results/plain_final_s32_33_34.npz", res=np.array(res, dtype=object))
    print("saved results/plain_final_s32_33_34.npz")
