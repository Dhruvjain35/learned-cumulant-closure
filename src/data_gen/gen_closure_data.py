"""
Generate the learnability dataset for the closure map.

For exact open Tavis-Cummings trajectories (validated solver), at each sampled time and for
each distinct emitter triple (i,j,k), we record:
  INPUT  = local low-order moments among sites {i,j,k} and the cavity, plus detunings
           (exactly the information a second-order cumulant model has at that instant).
  TARGET = the exact 3rd connected cumulant kappa3(sp_i, sm_j, sz_k)  (Re, Im),
           which is what the second-order cumulant closure sets to zero.

Because the feature/target definition is identical for every triple and every M, an MLP on
this LOCAL feature vector is permutation-equivariant and transfers across M by construction.
We tag every sample with (M, sigma) so we can test M- and disorder-EXTRAPOLATION.
"""
import numpy as np
import qutip as qt
from exact_tc import build_ops, build_H, collapse_ops, draw_disorder

PARAMS = dict(wc=1.0, gcoll=0.8, kappa=0.1, gdown=0.02, gphi=0.02)
NC = {3: 8, 4: 8, 5: 6}          # Fock cutoffs (superradiance-converged for M<=4; M=5 uses 6, checked lighter)
TMAX, NT = 40.0, 60


def E(op, rho): return qt.expect(op, rho)


def conn2(A, B, rho): return E(A * B, rho) - E(A, rho) * E(B, rho)


def conn3(A, B, C, rho):
    a, b, c = E(A, rho), E(B, rho), E(C, rho)
    return E(A * B * C, rho) - a * conn2(B, C, rho) - b * conn2(A, C, rho) - c * conn2(A, B, rho) - a * b * c


def state_moments(rho, ops):
    """Precompute all lower moments needed for features, once per state."""
    a, sm, sp, sz = ops["a"], ops["sm"], ops["sp"], ops["sz"]
    M = ops["M"]
    sz_v = np.array([E(sz[i], rho).real for i in range(M)])
    sm_v = np.array([E(sm[i], rho) for i in range(M)])
    nc = E(a.dag() * a, rho).real
    alpha = E(a, rho)
    cpm = np.zeros((M, M), complex)   # c2(sp_i, sm_j)
    czz = np.zeros((M, M))            # c2(sz_i, sz_j)
    cmz = np.zeros((M, M), complex)   # c2(sm_i, sz_j)
    cav_sm = np.array([conn2(a.dag(), sm[i], rho) for i in range(M)])  # c2(a^dag, sm_i)
    for i in range(M):
        for j in range(M):
            if i != j:
                cpm[i, j] = conn2(sp[i], sm[j], rho)
                czz[i, j] = conn2(sz[i], sz[j], rho).real
                cmz[i, j] = conn2(sm[i], sz[j], rho)
    return dict(sz=sz_v, sm=sm_v, nc=nc, alpha=alpha, cpm=cpm, czz=czz, cmz=cmz, cav_sm=cav_sm)


def features_for_triple(mom, delta, i, j, k):
    ri = lambda z: [z.real, z.imag]
    f = [delta[i], delta[j], delta[k],
         mom["sz"][i], mom["sz"][j], mom["sz"][k],
         *ri(mom["sm"][i]), *ri(mom["sm"][j]), *ri(mom["sm"][k]),
         *ri(mom["cpm"][i, j]), *ri(mom["cpm"][i, k]), *ri(mom["cpm"][j, k]),
         *ri(mom["cpm"][j, i]), *ri(mom["cpm"][k, i]), *ri(mom["cpm"][k, j]),
         mom["czz"][i, j], mom["czz"][i, k], mom["czz"][j, k],
         *ri(mom["cmz"][i, k]), *ri(mom["cmz"][j, k]), *ri(mom["cmz"][k, i]), *ri(mom["cmz"][k, j]),
         mom["nc"], *ri(mom["alpha"]),
         *ri(mom["cav_sm"][i]), *ri(mom["cav_sm"][j]), *ri(mom["cav_sm"][k])]
    return f


def trajectory_samples(M, sigma, seed):
    p = PARAMS
    Nc = NC[M]
    g0 = p["gcoll"] / np.sqrt(M)
    ops = build_ops(M, Nc)
    w = draw_disorder(M, p["wc"], sigma, seed, "gauss")
    delta = w - p["wc"]
    H = build_H(ops, p["wc"], w, np.full(M, g0))
    c = collapse_ops(ops, kappa=p["kappa"], gdown=p["gdown"], gphi=p["gphi"])
    psi0 = qt.tensor(qt.basis(Nc, 0), *[qt.basis(2, 0) for _ in range(M)])
    ts = np.linspace(0, TMAX, NT)
    res = qt.mesolve(H, psi0, ts, c_ops=c, options={"progress_bar": False})
    sp, sm, sz = ops["sp"], ops["sm"], ops["sz"]
    triples = [(i, j, k) for i in range(M) for j in range(M) for k in range(M) if len({i, j, k}) == 3]
    X, Y = [], []
    for rho in res.states:
        mom = state_moments(rho, ops)
        for (i, j, k) in triples:
            X.append(features_for_triple(mom, delta, i, j, k))
            kap = conn3(sp[i], sm[j], sz[k], rho)
            Y.append([kap.real, kap.imag])
    return np.array(X), np.array(Y)


def main():
    sigmas = [0.0, 0.05, 0.10, 0.20, 0.30]
    seeds = [0, 1, 2]
    Xall, Yall, Mtag, Stag = [], [], [], []
    for M in [3, 4, 5]:
        for sigma in sigmas:
            for seed in seeds:
                X, Y = trajectory_samples(M, sigma, seed)
                Xall.append(X); Yall.append(Y)
                Mtag.append(np.full(len(X), M)); Stag.append(np.full(len(X), sigma))
                print(f"   M={M} sigma={sigma:.2f} seed={seed}: {len(X)} samples  |kap3|_max={np.abs(Y).max():.4f}")
    X = np.vstack(Xall); Y = np.vstack(Yall); Mt = np.concatenate(Mtag); St = np.concatenate(Stag)
    np.savez_compressed("results/closure_data.npz", X=X, Y=Y, M=Mt, sigma=St)
    print(f"\nsaved results/closure_data.npz  X={X.shape}  Y={Y.shape}")


if __name__ == "__main__":
    print("Generating closure learnability dataset (exact trajectories)...")
    main()
