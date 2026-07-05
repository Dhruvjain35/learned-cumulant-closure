"""
Exact large-M reference by quantum trajectories (mcsolve) on the SAME exact_tc model as the validated dense
engine (identical H and c_ops), so there is zero convention risk. Unlike a full density-matrix mesolve
(rho is (Nc*2^M)^2, OOM past M~9) or piqs (identical emitters only), mcsolve evolves pure states of dim
Nc*2^M and works for DISORDERED emitters up to M~12-13. Monte Carlo error ~ std/sqrt(ntraj) is reported so
comparisons are honest.

This module: (a) validates mcsolve against the exact mesolve at M=5 (must agree within MC error), and
(b) times it at M=12 to confirm feasibility.
"""
import os
import time
import numpy as np
import qutip as qt
from exact_tc import build_ops, build_H, collapse_ops, draw_disorder

WC, GCOLL, KAPPA, GDOWN, GPHI = 1.0, 0.8, 0.1, 0.02, 0.02
TMAX, NT = 40.0, 200


def mc_reference(M, Nc, w, g, ntraj, seed=12345, kappa=KAPPA, gdown=GDOWN, gphi=GPHI):
    """Seeded (deterministic) -> cache on disk; the cache stores w and is only reused on an exact match.
    kappa/gdown/gphi default to the module constants (backward-compatible); pass pocket values to reference
    the low-loss regime. Loss params are folded into the cache key so pocket runs never collide with the
    original fixed-param caches."""
    tag = f"_k{kappa:g}_gd{gdown:g}_gp{gphi:g}" if (kappa, gdown, gphi) != (KAPPA, GDOWN, GPHI) else ""
    cache = f"results/mcref_M{M}_Nc{Nc}_n{ntraj}_s{seed}{tag}.npz"
    if os.path.exists(cache):
        d = np.load(cache)
        if np.array_equal(d["w"], np.asarray(w, float)):
            return d["ts"], d["n"], d["z"], float(d["mcerr"])
    ts, n, z, mcerr = _mc_reference(M, Nc, w, g, ntraj, seed, kappa, gdown, gphi)
    np.savez_compressed(cache, ts=ts, n=n, z=z, mcerr=mcerr, w=np.asarray(w, float))
    return ts, n, z, mcerr


def _mc_reference(M, Nc, w, g, ntraj, seed=12345, kappa=KAPPA, gdown=GDOWN, gphi=GPHI):
    ops = build_ops(M, Nc)
    H = build_H(ops, WC, w, g)
    cops = collapse_ops(ops, kappa=kappa, gdown=gdown, gphi=gphi)
    psi0 = qt.tensor(qt.basis(Nc, 0), *[qt.basis(2, 0) for _ in range(M)])   # all excited (|0>=excited)
    n = ops["a"].dag() * ops["a"]
    zmean = sum(ops["sz"]) / M
    ts = np.linspace(0, TMAX, NT)
    opts = {"progress_bar": False, "map": "parallel"}
    res = qt.mcsolve(H, psi0, ts, c_ops=cops, e_ops=[n, zmean], ntraj=ntraj,
                     options=opts, seeds=seed)
    nmean = np.array(res.expect[0]).real
    zmean_t = np.array(res.expect[1]).real
    # MC standard error from run-to-run spread of the trajectory average
    try:
        nse = np.array(res.std_expect[0]).real / np.sqrt(ntraj)
        zse = np.array(res.std_expect[1]).real / np.sqrt(ntraj)
        mcerr = float(np.nanmax([np.nanmax(nse), np.nanmax(zse)]))
    except Exception:
        mcerr = float("nan")
    return ts, nmean, zmean_t, mcerr


def mesolve_reference(M, Nc, w, g):
    ops = build_ops(M, Nc)
    H = build_H(ops, WC, w, g)
    cops = collapse_ops(ops, kappa=KAPPA, gdown=GDOWN, gphi=GPHI)
    psi0 = qt.tensor(qt.basis(Nc, 0), *[qt.basis(2, 0) for _ in range(M)])
    n = ops["a"].dag() * ops["a"]
    zmean = sum(ops["sz"]) / M
    ts = np.linspace(0, TMAX, NT)
    e = qt.mesolve(H, psi0, ts, c_ops=cops, e_ops=[n, zmean], options={"progress_bar": False}).expect
    return ts, np.array(e[0]).real, np.array(e[1]).real


if __name__ == "__main__":
    print("(a) mcsolve vs exact mesolve at M=5 (disordered seed 7): must agree within MC error")
    w = draw_disorder(5, WC, 0.15, 7, "gauss")
    g = np.full(5, GCOLL / np.sqrt(5))
    _, nm, zm, mcerr = mc_reference(5, 11, w, g, ntraj=400)
    _, ne, ze = mesolve_reference(5, 11, w, g)
    dn = float(np.max(np.abs(nm - ne)))
    dz = float(np.max(np.abs(zm - ze)))
    print(f"    max|n_mc - n_exact|={dn:.4f}  max|z_mc - z_exact|={dz:.4f}  reported MC err={mcerr:.4f}")
    print(f"    consistent = {dn < 5 * max(mcerr, 1e-3) and dz < 5 * max(mcerr, 1e-3)}\n")

    print("(b) mcsolve timing at M=12 (disordered), ntraj=200:")
    w = draw_disorder(12, WC, 0.15, 7, "gauss")
    g = np.full(12, GCOLL / np.sqrt(12))
    t0 = time.time()
    _, nm, zm, mcerr = mc_reference(12, 12, w, g, ntraj=200)
    print(f"    done in {time.time()-t0:.0f}s   n_peak={nm.max():.3f}  z_end={zm[-1]:.3f}  MC err={mcerr:.4f}")
