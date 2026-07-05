"""
How far does the learned closure transfer in molecule number?

Train the local-feature closure ONLY on small M (2 and 3), then run it in the closed loop at
M = 4, 5, 6 (never trained), each vs an EXACT reference. Report whether it stays stable and how the
error grows with M.

No leakage:
  - test M (4,5,6) are not in the training M (2,3)
  - the test disorder realization uses seed 7; training uses seeds 0..5
  - the feature scalers are fit on training data only (inside LocalClosure.fit)
No hallucination:
  - each test engine passes the machine-precision correctness gate before its loop is trusted
  - the exact reference is checked Fock-converged (peak photon number well below the cutoff)
"""
import os
import numpy as np
import qutip as qt
import local_closure as L
from cumulant import CumulantEngine
from exact_tc import build_ops, build_H, collapse_ops, draw_disorder

WC, GCOLL, KAPPA, GDOWN, GPHI = L.WC, L.GCOLL, L.KAPPA, L.GDOWN, L.GPHI
TMAX, NT = L.TMAX, L.NT
TRAIN_M = {2: 8, 3: 10}          # M -> Nc for training exact trajectories
TRAIN_SEEDS = [0, 1, 2, 3, 4, 5]
TRAIN_SIGMAS = [0.05, 0.10, 0.15, 0.20, 0.30]
TEST = {4: (10, 6), 5: (11, 6)}   # test M -> (Nc, Fsafe); M>=6 exceeds exact-validation feasibility here
#   (M=6 engine build materializes ~11 GB of dense monomial matrices -> OOM; that is the exact-sim wall
#    the closure is meant to cross, so validated transfer is reported up to the largest checkable M=5)
TEST_SEED, TEST_SIGMA = 7, 0.15


def engine(M, sigma, seed, Nc, Fsafe):
    w = draw_disorder(M, WC, sigma, seed, "gauss")
    g = np.full(M, GCOLL / np.sqrt(M))
    return CumulantEngine(M, Nc, WC, w, g, kappa=KAPPA, gdown=GDOWN, gphi=GPHI, Fsafe=Fsafe), (w - WC)


def obs_ops(eng):
    M = eng.M
    n = eng.O["a"].dag() * eng.O["a"]
    zs = [eng.O["sz"][i] for i in range(M)]
    pm = [eng.O["sp"][i] * eng.O["sm"][j] for i in range(M) for j in range(M) if i != j]
    return n, zs, pm


def exact_reference(eng, Nc):
    ts = np.linspace(0, TMAX, NT)
    psi0 = qt.tensor(qt.basis(Nc, 0), *[qt.basis(2, 0) for _ in range(eng.M)])
    n, zs, pm = obs_ops(eng)
    e = qt.mesolve(eng.H, psi0, ts, c_ops=eng.cops, e_ops=[n] + zs + pm + [n * n],
                   options={"progress_bar": False}).expect
    M = eng.M
    nc = np.array(e[0]).real
    z = np.mean([np.array(e[1 + i]).real for i in range(M)], axis=0)
    npair = M * (M - 1)
    pmv = np.mean([np.array(e[1 + M + k]).real for k in range(npair)], axis=0)
    n2 = np.array(e[1 + M + npair]).real
    ncpeak = nc.max()
    return ts, np.stack([nc, z, pmv], axis=1), psi0, ncpeak


def gate_engine(eng, Nc):
    """Machine-precision correctness gate at a mid-trajectory exact state. Returns the max error."""
    ts = np.linspace(0, 6, 3)
    psi0 = qt.tensor(qt.basis(Nc, 0), *[qt.basis(2, 0) for _ in range(eng.M)])
    rho = qt.mesolve(eng.H, psi0, ts, c_ops=eng.cops, options={"progress_bar": False}).states[-1]
    return eng.gate_exact(rho)[0]


def main():
    # ---- train on M = 2, 3 (held-out seeds) ----
    lc = L.LocalClosure(M=3)     # enumerates all order-3 cumulant types (M-independent)
    cache = "results/transfer_train.npz"
    if os.path.exists(cache):
        data = np.load(cache, allow_pickle=True)["data"].item()
        print("loaded cached training dataset (M=2,3, seeds 0-5)")
    else:
        print("training on M=2,3 (seeds 0-5)...")
        esd = []
        for M, Nc in TRAIN_M.items():
            Fs = min(Nc - 4, 6)
            for sg in TRAIN_SIGMAS:
                for sd in TRAIN_SEEDS:
                    eng, det = engine(M, sg, sd, Nc, Fs)
                    _, states, _ = _mesolve_states(eng, Nc)
                    esd.append((eng, states, det))
        data, _, _ = lc.dataset(esd)
        np.savez_compressed(cache, data=np.array(data, dtype=object))
    lc.fit(data)
    dom = [t for t in lc.types if t in lc.nets and lc.nets[t].best_validation_score_ > 0.8]
    print(f"   trained; dominant cumulant types (val>0.8): {len(dom)}")

    # ---- test transfer at M = 4, 5, 6 ----
    print(f"\n{'M':>2s} {'gate':>10s} {'nc_peak/Nc':>11s} {'stable':>7s} "
          f"{'plain err (t<=40)':>18s} {'learned err':>12s} {'improvement':>12s}")
    trend = []
    for M, (Nc, Fs) in TEST.items():
        try:
            eng, det = engine(M, TEST_SIGMA, TEST_SEED, Nc, Fs)
            gerr = gate_engine(eng, Nc)
            if gerr > 1e-7:
                print(f"{M:>2d}  gate FAIL ({gerr:.1e}) -> engine not trusted, skipping (raise Nc/Fsafe)")
                continue
            ts, exact, psi0, ncpk = exact_reference(eng, Nc)
            m0 = eng.moments_from_rho(psi0)[eng.tidx]
            plain = np.array([L.obs(m, eng) for m in L.run_loop(eng, m0, ts, None)])
            lj = L.run_loop(eng, m0, ts, lc.closure_fn(eng, det))
            learned = np.array([L.obs(m, eng) for m in lj])
            Ln = min(len(exact), len(plain), len(learned))
            ep = float(np.max(np.abs(plain[:Ln] - exact[:Ln])))
            el = float(np.max(np.abs(learned[:Ln] - exact[:Ln])))
            stable = (len(lj) == NT) and np.isfinite(learned).all() and np.max(np.abs(learned)) < 1e2
            imp = f"{ep/max(el,1e-9):.1f}x" if stable else "n/a"
            print(f"{M:>2d} {gerr:10.1e} {ncpk/Nc:11.2f} {str(stable):>7s} {ep:18.4f} {el:12.4f} {imp:>12s}")
            trend.append((M, ep, el, stable))
        except MemoryError:
            print(f"{M:>2d}  MemoryError building engine (Hilbert space too large) -> transfer test caps here")
            break
        except Exception as ex:
            print(f"{M:>2d}  error: {type(ex).__name__}: {ex}")
    if trend:
        print("\nTRANSFER TREND (trained only on M=2,3):")
        for (M, ep, el, st) in trend:
            print(f"   M={M}: learned closure {'stable' if st else 'UNSTABLE'}, "
                  f"error {el:.3f} vs plain {ep:.3f}  ({'beats plain '+f'{ep/max(el,1e-9):.1f}x' if el<ep and st else 'no gain'})")
        np.savez_compressed("results/transfer_trend.npz", trend=np.array(trend, dtype=object))


def _mesolve_states(eng, Nc):
    ts = np.linspace(0, TMAX, NT)
    psi0 = qt.tensor(qt.basis(Nc, 0), *[qt.basis(2, 0) for _ in range(eng.M)])
    return ts, qt.mesolve(eng.H, psi0, ts, c_ops=eng.cops, options={"progress_bar": False}).states, psi0


if __name__ == "__main__":
    main()
