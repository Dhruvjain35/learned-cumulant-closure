"""
Generate EXACT closure training data at M=4,5 WITHOUT the dense cumulant engine.

Why: Result 8 showed the closure trained only on M=2,3 loses its edge by M~10. The fix is to retrain on
larger M. The old data path needed the dense CumulantEngine (13 min build at M=5, OOM at M=6) only to
evaluate monomial expectations; here we instead pass every monomial operator (order <= 3, enumerated
sparsely) as e_ops to mesolve, which returns the exact moments directly. No projection, no dense engine.

Protocol (identical leakage rules as before):
  - training seeds 0..3 only; the test disorder seed 7 is NEVER touched
  - training sigmas {0.05, 0.15, 0.30}
  - targets: kappa3(k) = <k> - [2nd-order cumulant factorization of <k>]  (exact, complex)
  - features: identical code path as the trained closure (feature_plan/extract + LocalClosure.context)
  - Fock safety: nc_peak/Nc printed per run and required < 0.5

Output: results/transfer_train_v2.npz -- the OLD M=2,3 dataset merged with the new M=4,5 samples, same
{type: (X, Y)} format, so LocalClosure.fit() consumes it unchanged.
"""
import os
import time
import numpy as np
import qutip as qt
from cumulant import mono_matrix, mono_order, closure_terms
from scalable_engine import enum_sparse
from local_closure import LocalClosure, feature_plan, extract
from exact_tc import build_ops, build_H, collapse_ops, draw_disorder

WC, GCOLL, KAPPA, GDOWN, GPHI = 1.0, 0.8, 0.1, 0.02, 0.02
TMAX, NT = 40.0, 200
GEN = {4: 10, 5: 11}                  # M -> Nc (same cutoffs that were Fock-verified in transfer_scan)
SEEDS = [0, 1, 2, 3]
SIGMAS = [0.05, 0.15, 0.30]
TSTRIDE = 2                           # sample every 2nd time point
OLD_CACHE = "results/transfer_train.npz"
OUT = "results/transfer_train_v2.npz"


def moments_by_eops(M, Nc, w, g):
    """Exact moments of every order<=3 monomial along the superradiant trajectory, via mesolve e_ops.
    Returns (keys, {key: complex array over time}, nc_peak)."""
    ops = build_ops(M, Nc)
    H = build_H(ops, WC, w, g)
    cops = collapse_ops(ops, kappa=KAPPA, gdown=GDOWN, gphi=GPHI)
    psi0 = qt.tensor(qt.basis(Nc, 0), *[qt.basis(2, 0) for _ in range(M)])   # |0> = excited (qutip sigma conv.)
    keys = enum_sparse(M, 3)
    e_ops = [mono_matrix(k, ops) for k in keys]
    ts = np.linspace(0, TMAX, NT)
    res = qt.mesolve(H, psi0, ts, c_ops=cops, e_ops=e_ops, options={"progress_bar": False})
    # keep COMPLEX values: monomials like a, s+ are non-Hermitian and their moments are complex
    moms = {k: np.asarray(res.expect[i], dtype=complex) for i, k in enumerate(keys)}
    nc_peak = float(np.max(moms[(1, 1, tuple([0] * M))].real))
    return keys, moms, nc_peak


def add_samples(M, keys, moms, det, Xd, Yd):
    """Append (features, kappa3) samples for every order-3 monomial at strided time points.
    Identical math/features to LocalClosure.dataset: kappa3 = <k> - factorized, features via extract()."""
    o3 = [k for k in keys if mono_order(k) == 3]
    plans = {k: feature_plan(k) for k in o3}
    for ti in range(0, NT, TSTRIDE):
        mdict = {k: moms[k][ti] for k in keys}
        ctx = LocalClosure.context(None, mdict, M)          # unbound call: provably the same feature code
        for k in o3:
            terms, _ = closure_terms(k)
            val = 0j
            for coeff, kk in terms:
                prod = coeff
                for f in kk:
                    prod *= mdict[f]
                val += prod
            kap = mdict[k] - val                             # exact 3rd cumulant (complex)
            sig = plans[k][0]
            Xd.setdefault(sig, []).append(extract(mdict, plans[k], det, ctx))
            Yd.setdefault(sig, []).append([kap.real, kap.imag])


if __name__ == "__main__":
    Xd, Yd = {}, {}
    for M, Nc in GEN.items():
        for sg in SIGMAS:
            for sd in SEEDS:
                t0 = time.time()
                w = draw_disorder(M, WC, sg, sd, "gauss")
                g = np.full(M, GCOLL / np.sqrt(M))
                keys, moms, nc_peak = moments_by_eops(M, Nc, w, g)
                assert nc_peak / Nc < 0.5, f"Fock cutoff too tight: nc_peak/Nc={nc_peak/Nc:.2f}"
                add_samples(M, keys, moms, w - WC, Xd, Yd)
                print(f"M={M} sigma={sg:.2f} seed={sd}: nc_peak/Nc={nc_peak/Nc:.2f}  {time.time()-t0:.0f}s",
                      flush=True)

    new = {t: (np.array(Xd[t]), np.array(Yd[t])) for t in Xd}
    old = np.load(OLD_CACHE, allow_pickle=True)["data"].item()
    merged = {}
    for t in sorted(set(old) | set(new)):
        Xs = [d[t][0] for d in (old, new) if t in d]
        Ys = [d[t][1] for d in (old, new) if t in d]
        merged[t] = (np.vstack(Xs), np.vstack(Ys))
    np.savez_compressed(OUT, data=np.array(merged, dtype=object))
    tot_old = sum(len(v[0]) for v in old.values())
    tot_new = sum(len(v[0]) for v in new.values())
    print(f"\nsaved {OUT}: {tot_old} old (M=2,3) + {tot_new} new (M=4,5) samples")
