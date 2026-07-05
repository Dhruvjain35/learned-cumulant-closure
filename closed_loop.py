"""
Closed-loop cumulant dynamics. Integrate the (validated) second-order cumulant EOM and compare
to exact, for open disordered Tavis-Cummings superradiance:

  exact            : full Lindblad (QuTiP)
  cumulant (kap3=0): plain 2nd-order cumulant  -> shows the gap the closure must fill
  cumulant (oracle): kap3 taken from exact at each time -> must reproduce exact (integrator sanity)

Observables: n=<a^dag a>, mean <sigma^z>, mean_{i!=j} Re<sigma_i^+ sigma_j^->.
"""
import numpy as np
import qutip as qt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from cumulant import CumulantEngine, mono_matrix, mono_order

M = 3
NC = 12
FSAFE = 6
WC = 1.0
GCOLL = 0.8
KAPPA, GDOWN, GPHI = 0.1, 0.02, 0.02
SIGMA = 0.15
SEED = 1
TMAX, NT = 40.0, 200


def build(sigma, seed):
    from exact_tc import draw_disorder
    w = draw_disorder(M, WC, sigma, seed, "gauss")
    g = np.full(M, GCOLL / np.sqrt(M))
    eng = CumulantEngine(M, NC, WC, w, g, kappa=KAPPA, gdown=GDOWN, gphi=GPHI, Fsafe=FSAFE)
    return eng


def keys_for_obs(eng):
    z = [(0, 0, tuple(3 if s == i else 0 for s in range(M))) for i in range(M)]      # <sz_i>
    nkey = (1, 1, tuple([0] * M))                                                     # <a^dag a>
    pm = [(0, 0, tuple({i: 1, j: 2}.get(s, 0) for s in range(M)))                     # <s+_i s-_j>
          for i in range(M) for j in range(M) if i != j]
    return nkey, z, pm


def obs_from_track(mtrack, eng, nkey, zk, pmk):
    ti = {k: a for a, k in enumerate(eng.track)}
    n = mtrack[ti[nkey]].real
    z = np.mean([mtrack[ti[k]].real for k in zk])
    pm = np.mean([mtrack[ti[k]].real for k in pmk])
    return n, z, pm


def pack(m): return np.concatenate([m.real, m.imag])
def unpack(y): n = len(y) // 2; return y[:n] + 1j * y[n:]


def integrate_cumulant(eng, m0, ts, kap3_of_t=None):
    def rhs(t, y):
        m = unpack(y)
        kap3 = kap3_of_t(t) if kap3_of_t else {}
        return pack(eng.deriv(m, kap3))
    sol = solve_ivp(rhs, (ts[0], ts[-1]), pack(m0), t_eval=ts, rtol=1e-7, atol=1e-9, method="RK45")
    return np.array([unpack(sol.y[:, i]) for i in range(sol.y.shape[1])])


def main():
    eng = build(SIGMA, SEED)
    ops = eng.O
    nkey, zk, pmk = keys_for_obs(eng)
    ts = np.linspace(0, TMAX, NT)
    psi0 = qt.tensor(qt.basis(NC, 0), *[qt.basis(2, 0) for _ in range(M)])

    # exact trajectory (states, so we can also extract exact kap3 for the oracle)
    res = qt.mesolve(eng.H, psi0, ts, c_ops=eng.cops, options={"progress_bar": False})
    exact_obs = np.array([obs_from_track(eng.moments_from_rho(r)[eng.tidx], eng, nkey, zk, pmk) for r in res.states])

    # initial tracked moments
    m0 = eng.moments_from_rho(psi0)[eng.tidx]

    # (a) plain 2nd-order cumulant  (kap3 = 0)
    traj0 = integrate_cumulant(eng, m0, ts, None)
    plain_obs = np.array([obs_from_track(m, eng, nkey, zk, pmk) for m in traj0])

    # (b) oracle closure: kap3(t) from exact states, interpolated
    o3 = eng.order3
    base_of = []
    kap3_series = []
    for r in res.states:
        mfull = eng.moments_from_rho(r)
        mt = mfull[eng.tidx]
        base = eng._full_from_tracked(mt, {})
        kap3_series.append(np.array([mfull[eng.idx[k]] - base[eng.idx[k]] for k in o3]))
    kap3_series = np.array(kap3_series)
    interp_re = interp1d(ts, kap3_series.real, axis=0, bounds_error=False, fill_value="extrapolate")
    interp_im = interp1d(ts, kap3_series.imag, axis=0, bounds_error=False, fill_value="extrapolate")

    def oracle(t):
        v = interp_re(t) + 1j * interp_im(t)
        return {k: v[i] for i, k in enumerate(o3)}

    trajO = integrate_cumulant(eng, m0, ts, oracle)
    oracle_obs = np.array([obs_from_track(m, eng, nkey, zk, pmk) for m in trajO])

    # report
    labels = ["<a^dag a>", "mean<sz>", "mean Re<s+ s->"]
    print(f"\nM={M}, sigma={SIGMA}, gcoll={GCOLL}, kappa={KAPPA}  (superradiance)")
    print(f"{'observable':16s} {'plain-cumulant max err':>24s} {'oracle-closure max err':>24s}")
    for c in range(3):
        e_plain = np.max(np.abs(plain_obs[:, c] - exact_obs[:, c]))
        e_oracle = np.max(np.abs(oracle_obs[:, c] - exact_obs[:, c]))
        print(f"{labels[c]:16s} {e_plain:24.4f} {e_oracle:24.6f}")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(13, 4))
        for c in range(3):
            ax[c].plot(ts, exact_obs[:, c], "k-", lw=2, label="exact")
            ax[c].plot(ts, plain_obs[:, c], "r--", lw=1.5, label="plain cumulant")
            ax[c].plot(ts, oracle_obs[:, c], "g:", lw=1.5, label="oracle closure")
            ax[c].set_title(labels[c]); ax[c].set_xlabel("t"); ax[c].grid(alpha=.3); ax[c].legend()
        fig.suptitle(f"Closed-loop cumulant vs exact (M={M}, sigma={SIGMA}, superradiance)")
        fig.tight_layout(); fig.savefig("results/closed_loop.png", dpi=130)
        print("\nsaved results/closed_loop.png")
    except Exception as e:
        print("plot skipped:", e)
    return ts, exact_obs, plain_obs, oracle_obs


if __name__ == "__main__":
    main()
