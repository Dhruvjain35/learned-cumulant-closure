"""
Regime-grid training data for the v3 closure: exact moments across a grid of physical regimes at M=2..5.

Everything the v2 generator proved (e_ops path == dense path exactly, check_eops_path.py) carries over;
this adds the regime grid and the v3 features. Leakage RULE unchanged (test seeds >= 7 and test M=8,10
never touched); the training seed set here is {0,1,2}, DECORRELATED across sigma: draw_disorder returns
the same standard-normal direction rescaled when (seed) is fixed and only sigma changes, which would give
just 3 detuning directions per M across the whole grid. We therefore derive the rng seed as
seed*100 + round(sigma*100)  (values {5,15,30,105,115,130,205,215,230} -- disjoint from test seeds 7,8,9),
giving 9 distinct detuning directions per M. Directions are shared across M (rng-prefix property of
draw_disorder); that is deliberate and recorded, not hidden.

Grid (162 configs per M):
  gcoll in {0.4, 0.7, 1.0}   kappa in {0.05, 0.15, 0.3}   gphi in {0.02, 0.1}
  sigma in {0.05, 0.15, 0.3}  disorder seeds {0, 1, 2}     gdown = 0.02 fixed, gpump = 0
TSTRIDE=8 (25 time points per trajectory) keeps totals near the fit budget; per-type sample cap 600k
(seeded subsample) bounds MLP fit time.

Fock safety per run: nc_peak/Nc < 0.5 required. A violating config is SKIPPED AND LOGGED (not a crash:
one hot config must not kill a 2-hour campaign), and the skip list is saved with the dataset -- silent
truncation would misrepresent coverage.
"""
import time
import numpy as np
import qutip as qt
from cumulant import mono_matrix
from scalable_engine import enum_sparse
from closure_v3 import add_samples_v3
from exact_tc import build_ops, build_H, collapse_ops, draw_disorder

WC, GDOWN = 1.0, 0.02
TMAX, NT, TSTRIDE = 40.0, 200, 8
NCS = {2: 9, 3: 10, 4: 11, 5: 12}
GCOLLS = [0.4, 0.7, 1.0]
KAPPAS = [0.05, 0.15, 0.3]
GPHIS = [0.02, 0.1]
SIGMAS = [0.05, 0.15, 0.3]
SEEDS = [0, 1, 2]
CAP = 600_000
OUT = "results/train_v3.npz"


def train_rng_seed(sd, sg):
    """Decorrelate disorder direction across sigma (see module docstring). Disjoint from test seeds >=7."""
    return sd * 100 + int(round(sg * 100))


def moments_v3(M, Nc, w, g, kappa, gphi, eops_cache):
    """Exact moments of every order<=3 monomial via mesolve e_ops. e_ops depend only on (M,Nc) -> cached."""
    if M not in eops_cache:
        ops = build_ops(M, Nc)
        keys = enum_sparse(M, 3)
        eops_cache[M] = (ops, keys, [mono_matrix(k, ops) for k in keys])
    ops, keys, e_ops = eops_cache[M]
    H = build_H(ops, WC, w, g)
    cops = collapse_ops(ops, kappa=kappa, gdown=GDOWN, gphi=gphi)
    psi0 = qt.tensor(qt.basis(Nc, 0), *[qt.basis(2, 0) for _ in range(M)])   # |0> = excited
    ts = np.linspace(0, TMAX, NT)
    res = qt.mesolve(H, psi0, ts, c_ops=cops, e_ops=e_ops, options={"progress_bar": False})
    moms = {k: np.asarray(res.expect[i], dtype=complex) for i, k in enumerate(keys)}
    nc_peak = float(np.max(moms[(1, 1, tuple([0] * M))].real))
    return keys, moms, nc_peak


def config_list(pilot):
    """Full grid, or the pre-registered pilot subset (PROTOCOL.md): corners+center of (gcoll,kappa) x both
    gphi x corner sigmas, seed 0 only, M={2,3,4}; M=5 only at the (gcoll,kappa) corners, sigma=0.15."""
    cfgs = []
    if not pilot:
        for M, Nc in NCS.items():
            for gc in GCOLLS:
                for ka in KAPPAS:
                    for gp in GPHIS:
                        for sg in SIGMAS:
                            for sd in SEEDS:
                                cfgs.append((M, Nc, gc, ka, gp, sg, sd))
        return cfgs
    corners = [(0.4, 0.05), (0.4, 0.3), (1.0, 0.05), (1.0, 0.3), (0.7, 0.15)]
    for M in (2, 3, 4):
        for gc, ka in corners:
            for gp in GPHIS:
                for sg in (0.05, 0.3):
                    cfgs.append((M, NCS[M], gc, ka, gp, sg, 0))
    for gc, ka in [(0.4, 0.05), (0.4, 0.3), (1.0, 0.05), (1.0, 0.3)]:
        for gp in GPHIS:
            cfgs.append((5, NCS[5], gc, ka, gp, 0.15, 0))
    return cfgs


if __name__ == "__main__":
    import sys
    pilot = "--pilot" in sys.argv
    out = "results/train_v3_pilot.npz" if pilot else OUT
    # asserted, not eyeballed: every (seed, sigma) pair -> unique rng seed, disjoint from test seeds.
    # (M deliberately shares the direction at fixed (seed, sigma) -- rng-prefix property, documented above.)
    TEST_SEEDS = {7, 8, 9}
    seed_map = {(sd, sg): train_rng_seed(sd, sg) for sd in SEEDS for sg in SIGMAS}
    assert len(set(seed_map.values())) == len(seed_map), f"rng-seed collision: {seed_map}"
    assert not set(seed_map.values()) & TEST_SEEDS, f"train rng seeds hit test seeds: {seed_map}"
    Xd, Yd = {}, {}
    skipped = []
    eops_cache = {}
    cfgs = config_list(pilot)
    total = len(cfgs)
    for done, (M, Nc, gc, ka, gp, sg, sd) in enumerate(cfgs, 1):
        t0 = time.time()
        rs = train_rng_seed(sd, sg)
        w = draw_disorder(M, WC, sg, rs, "gauss")
        g = np.full(M, gc / np.sqrt(M))
        keys, moms, nc_peak = moments_v3(M, Nc, w, g, ka, gp, eops_cache)
        tag = f"M={M} gc={gc} ka={ka} gp={gp} sg={sg} sd={sd}(rng {rs})"
        if nc_peak / Nc >= 0.5:
            skipped.append(tag)
            print(f"[{done}/{total}] SKIP (nc_peak/Nc={nc_peak/Nc:.2f}) {tag}", flush=True)
            continue
        add_samples_v3(M, keys, lambda ti: {k: moms[k][ti] for k in keys},
                       range(0, NT, TSTRIDE), w - WC, (gc, ka, gp), Xd, Yd)
        print(f"[{done}/{total}] {tag}: nc_peak/Nc={nc_peak/Nc:.2f}  {time.time()-t0:.0f}s", flush=True)

    rng = np.random.default_rng(0)
    data = {}
    for t in Xd:
        X, Y = np.array(Xd[t]), np.array(Yd[t])
        if len(X) > CAP:
            idx = rng.choice(len(X), CAP, replace=False)
            X, Y = X[idx], Y[idx]
        data[t] = (X, Y)
    np.savez_compressed(out, data=np.array(data, dtype=object),
                        skipped=np.array(skipped, dtype=object),
                        seed_map=np.array(sorted((k, v) for k, v in seed_map.items()), dtype=object))
    print(f"\nsaved {out}: {sum(len(v[0]) for v in data.values())} samples across {len(data)} types; "
          f"{len(skipped)} configs skipped", flush=True)
    if skipped:
        print("skipped configs (Fock-unsafe at this Nc):")
        for s in skipped:
            print("   " + s)
