"""
Pocket-local training density (arm 2/4 of the 2x2 fix experiment): exact training data INSIDE the
observed instability pocket -- kappa in {0.03, 0.04, 0.05} x gphi in {0.035, 0.05, 0.065} at g=0.7,
sigma in {0.05, 0.15, 0.3}, training seeds {0,1,2} (sigma-decorrelated as in gen_data_v3), M=2..5.
Judgment seeds for the 2x2 (9, 10) remain untouched, as do the diagnostic seeds (7, 8).
Output: results/train_pocket.npz (same {type: (X,Y)} format; merged with train_v3.npz at fit time).
"""
import time
import numpy as np
from gen_data_v3 import moments_v3, train_rng_seed, NCS, WC, NT, TSTRIDE
from closure_v3 import add_samples_v3
from exact_tc import draw_disorder

GC = 0.7
KAS = [0.03, 0.04, 0.05]
GPS = [0.035, 0.05, 0.065]
SIGMAS = [0.05, 0.15, 0.3]
SEEDS = [0, 1, 2]
OUT = "results/train_pocket.npz"

if __name__ == "__main__":
    Xd, Yd = {}, {}
    eops_cache = {}
    skipped = []
    cfgs = [(M, NCS[M], ka, gp, sg, sd) for M in NCS for ka in KAS for gp in GPS
            for sg in SIGMAS for sd in SEEDS]
    for done, (M, Nc, ka, gp, sg, sd) in enumerate(cfgs, 1):
        t0 = time.time()
        rs = train_rng_seed(sd, sg)
        w = draw_disorder(M, WC, sg, rs, "gauss")
        g = np.full(M, GC / np.sqrt(M))
        keys, moms, nc_peak = moments_v3(M, Nc, w, g, ka, gp, eops_cache)
        tag = f"M={M} ka={ka} gp={gp} sg={sg} sd={sd}(rng {rs})"
        if nc_peak / Nc >= 0.5:
            skipped.append(tag)
            print(f"[{done}/{len(cfgs)}] SKIP (nc_peak/Nc={nc_peak/Nc:.2f}) {tag}", flush=True)
            continue
        add_samples_v3(M, keys, lambda ti: {k: moms[k][ti] for k in keys},
                       range(0, NT, TSTRIDE), w - WC, (GC, ka, gp), Xd, Yd)
        print(f"[{done}/{len(cfgs)}] {tag}: nc_peak/Nc={nc_peak/Nc:.2f}  {time.time()-t0:.0f}s", flush=True)
    data = {t: (np.array(Xd[t]), np.array(Yd[t])) for t in Xd}
    np.savez_compressed(OUT, data=np.array(data, dtype=object), skipped=np.array(skipped, dtype=object))
    print(f"\nsaved {OUT}: {sum(len(v[0]) for v in data.values())} samples; {len(skipped)} skipped")
