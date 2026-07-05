"""
V6 data generation: regenerate ALL training configs (full regime grid, 648, from gen_data_v3.config_list +
pocket set, 324, from gen_pocket_density's grid) with 27-wide trace-augmented features AND per-sample
provenance, so every training mix (V6 balanced / V5 balanced-no-traces = cols 0:21 / V4' count-matched
natural mix) is DERIVED at fit time with exact logged per-(kappa,gphi)-point contributions.

Leakage rules: training rng seeds from train_rng_seed(sd, sg) with sd in {0,1,2} give the derived set
{5,15,30,105,115,130,205,215,230}. REVIEW-CAUGHT COLLISION: derived seed 15 equals former ledger seed 15
(V5 dev) -- same rng stream at sigma=0.15 means an identical disorder realization. ADJUDICATION (dated in
PROTOCOL.md): ledger seed 15 is replaced by 24; the disjointness of the derived set against the FULL
amended ledger is ASSERTED below, not eyeballed. Same Fock guard (skip-and-log). Traces per config use
that config's own physical rates: lambda = (kappa, gdown, gdown/2 + gphi) for (nc/M, z, coh).

Output results/train_v6.npz:
  data:   {type: (X 27-wide, Y, P point-ids)}
  points: array of (region, M, gc, ka, gp, sg, sd) rows indexed by point-id
"""
import time
import numpy as np
from gen_data_v3 import (moments_v3, train_rng_seed, config_list, NCS, WC, NT, TSTRIDE,
                         GPHIS as MAIN_GPHIS)  # noqa: F401
import gen_pocket_density as gp_mod
from closure_v3 import train_canonical, context_v3
from closure_v6 import lambdas, trace_series, trace_features, extract_v6
from cumulant import mono_order, closure_terms
from local_closure import feature_plan
from exact_tc import draw_disorder

OUT = "results/train_v6.npz"
DT = 40.0 / (NT - 1)


def all_configs():
    cfgs = [("main", M, Nc, gc, ka, gp, sg, sd) for (M, Nc, gc, ka, gp, sg, sd) in config_list(False)]
    for M, Nc in NCS.items():
        for ka in gp_mod.KAS:
            for gpv in gp_mod.GPS:
                for sg in gp_mod.SIGMAS:
                    for sd in gp_mod.SEEDS:
                        cfgs.append(("pocket", M, Nc, gp_mod.GC, ka, gpv, sg, sd))
    return cfgs


def add_samples_v6(M, keys, moms, det, regime, lam, pid, Xd, Yd, Pd):
    """Trace-augmented samples for canonical order-3 monomials at strided time points."""
    ctx_series = np.array([context_v3({k: moms[k][ti] for k in keys}, M) for ti in range(NT)])
    h_series = trace_series(ctx_series, lam, DT)
    o3 = [k for k in keys if mono_order(k) == 3 and train_canonical(k)]
    plans = {k: feature_plan(k) for k in o3}
    for ti in range(0, NT, TSTRIDE):
        mdict = {k: moms[k][ti] for k in keys}
        ctx = list(ctx_series[ti])
        tr = trace_features(ctx_series[ti], h_series[ti])
        for k in o3:
            terms, _ = closure_terms(k)
            val = 0j
            for coeff, kk in terms:
                prod = coeff
                for f in kk:
                    prod *= mdict[f]
                val += prod
            kap = mdict[k] - val
            sig = plans[k][0]
            Xd.setdefault(sig, []).append(extract_v6(mdict, plans[k], det, ctx, regime, tr))
            Yd.setdefault(sig, []).append([kap.real, kap.imag])
            Pd.setdefault(sig, []).append(pid)


if __name__ == "__main__":
    # binding disjointness check: derived training rng seeds vs the FULL amended ledger (PROTOCOL.md 4/4b:
    # dev+verdict seeds 7..14, 16..24 -- seed 15 retired due to the collision, replaced by 24)
    LEDGER = set(range(7, 15)) | set(range(16, 25))
    derived = {train_rng_seed(sd, sg) for sd in (0, 1, 2) for sg in (0.05, 0.15, 0.3)}
    assert not (derived & LEDGER), f"training rng seeds collide with ledger: {derived & LEDGER}"
    cfgs = all_configs()
    Xd, Yd, Pd = {}, {}, {}
    points, skipped = [], []
    eops_cache = {}
    for done, (region, M, Nc, gc, ka, gpv, sg, sd) in enumerate(cfgs, 1):
        t0 = time.time()
        rs = train_rng_seed(sd, sg)
        w = draw_disorder(M, WC, sg, rs, "gauss")
        g = np.full(M, gc / np.sqrt(M))
        keys, moms, nc_peak = moments_v3(M, Nc, w, g, ka, gpv, eops_cache)
        tag = f"{region} M={M} gc={gc} ka={ka} gp={gpv} sg={sg} sd={sd}"
        if nc_peak / Nc >= 0.5:
            skipped.append(tag)
            print(f"[{done}/{len(cfgs)}] SKIP (nc_peak/Nc={nc_peak/Nc:.2f}) {tag}", flush=True)
            continue
        pid = len(points)
        points.append((region, M, gc, ka, gpv, sg, sd))
        add_samples_v6(M, keys, moms, w - WC, (gc, ka, gpv), lambdas(ka, gpv), pid, Xd, Yd, Pd)
        print(f"[{done}/{len(cfgs)}] {tag}: nc_peak/Nc={nc_peak/Nc:.2f}  {time.time()-t0:.0f}s", flush=True)

    data = {t: (np.array(Xd[t]), np.array(Yd[t]), np.array(Pd[t], dtype=np.int32)) for t in Xd}
    np.savez_compressed(OUT, data=np.array(data, dtype=object),
                        points=np.array(points, dtype=object),
                        skipped=np.array(skipped, dtype=object))
    print(f"\nsaved {OUT}: {sum(len(v[0]) for v in data.values())} samples, "
          f"{len(points)} configs, {len(skipped)} skipped", flush=True)
