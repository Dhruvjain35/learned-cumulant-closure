"""
CAPSTONE: a LEARNED closure in the loop. Train a network to predict the full order-3 cumulant
vector kap3 from the tracked moments, then integrate the cumulant EOM with the closure supplied
LIVE by the network (kap3 = net(current moments)), on a HELD-OUT disorder realization. This tests
the panel's central risk: is a learned closure stable in feedback, and does it beat plain cumulant?

Fixed M=3 here (the network input is the M=3 moment vector). Cross-M transfer of the closure map is
evidenced separately by learn_closure.py (train M{3,4} -> predict M5 at R2=0.71).
"""
import numpy as np
import qutip as qt
from scipy.integrate import solve_ivp
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from cumulant import CumulantEngine
from exact_tc import draw_disorder

M, NC, FSAFE = 3, 12, 6
WC, GCOLL, KAPPA, GDOWN, GPHI = 1.0, 0.8, 0.1, 0.02, 0.02
TMAX, NT = 40.0, 200


def make_engine(sigma, seed):
    w = draw_disorder(M, WC, sigma, seed, "gauss")
    g = np.full(M, GCOLL / np.sqrt(M))
    return CumulantEngine(M, NC, WC, w, g, kappa=KAPPA, gdown=GDOWN, gphi=GPHI, Fsafe=FSAFE), w


def exact_traj(eng):
    ts = np.linspace(0, TMAX, NT)
    psi0 = qt.tensor(qt.basis(NC, 0), *[qt.basis(2, 0) for _ in range(M)])
    res = qt.mesolve(eng.H, psi0, ts, c_ops=eng.cops, options={"progress_bar": False})
    return ts, res.states, psi0


def sample_pairs(eng, states):
    """(tracked moments -> order-3 kap3) along an exact trajectory."""
    o3 = eng.order3
    X, Y = [], []
    for r in states:
        mfull = eng.moments_from_rho(r)
        mt = mfull[eng.tidx]
        base = eng._full_from_tracked(mt, {})
        kap3 = np.array([mfull[eng.idx[k]] - base[eng.idx[k]] for k in o3])
        X.append(np.concatenate([mt.real, mt.imag]))
        Y.append(np.concatenate([kap3.real, kap3.imag]))
    return np.array(X), np.array(Y)


def obs(mtrack, eng, nkey, zk, pmk):
    ti = eng._ti
    return (mtrack[ti[nkey]].real,
            np.mean([mtrack[ti[k]].real for k in zk]),
            np.mean([mtrack[ti[k]].real for k in pmk]))


def keys(eng):
    z = [(0, 0, tuple(3 if s == i else 0 for s in range(M))) for i in range(M)]
    nkey = (1, 1, tuple([0] * M))
    pm = [(0, 0, tuple({i: 1, j: 2}.get(s, 0) for s in range(M))) for i in range(M) for j in range(M) if i != j]
    return nkey, z, pm


def pack(m): return np.concatenate([m.real, m.imag])
def unpack(y): n = len(y) // 2; return y[:n] + 1j * y[n:]


def integrate(eng, m0, ts, kap3_fn):
    o3 = eng.order3

    def rhs(t, y):
        m = unpack(y)
        kap3 = kap3_fn(m) if kap3_fn else {}
        return pack(eng.deriv(m, kap3))
    sol = solve_ivp(rhs, (ts[0], ts[-1]), pack(m0), t_eval=ts, rtol=1e-6, atol=1e-8, method="RK45")
    return np.array([unpack(sol.y[:, i]) for i in range(sol.y.shape[1])])


def main():
    # ---- training data from several disorder realizations ----
    import os
    cache = "results/loop_train.npz"
    if os.path.exists(cache):
        d = np.load(cache); Xtr, Ytr = d["X"], d["Y"]
        print(f"loaded cached training data {Xtr.shape}")
    else:
        print("generating training data (exact trajectories)...")
        Xtr, Ytr = [], []
        for sigma in [0.05, 0.10, 0.15, 0.20, 0.30]:
            for seed in [0, 1, 2, 3, 4, 5]:
                eng, _ = make_engine(sigma, seed)
                _, states, _ = exact_traj(eng)
                X, Y = sample_pairs(eng, states)
                Xtr.append(X); Ytr.append(Y)
        Xtr = np.vstack(Xtr); Ytr = np.vstack(Ytr)
        np.savez_compressed(cache, X=Xtr, Y=Ytr)
    ylo, yhi = Ytr.min(0), Ytr.max(0)          # per-component range for an in-loop safeguard
    print(f"   train samples: {Xtr.shape}, targets {Ytr.shape}")
    sc = StandardScaler().fit(Xtr)
    net = MLPRegressor(hidden_layer_sizes=(256, 256), activation="relu", alpha=1e-5,
                       max_iter=1200, early_stopping=True, n_iter_no_change=25, random_state=0)
    net.fit(sc.transform(Xtr), Ytr)
    print(f"   closure network trained (val score ~ {net.best_validation_score_:.3f})")

    # ---- held-out disorder realization: run the loop ----
    eng, _ = make_engine(0.15, 7)          # seed 7 unseen
    eng._ti = {k: a for a, k in enumerate(eng.track)}
    nkey, zk, pmk = keys(eng)
    ts, states, psi0 = exact_traj(eng)
    exact = np.array([obs(eng.moments_from_rho(r)[eng.tidx], eng, nkey, zk, pmk) for r in states])
    m0 = eng.moments_from_rho(psi0)[eng.tidx]
    o3 = eng.order3

    def learned_closure(m):
        v = net.predict(sc.transform(pack(m)[None, :]))[0]
        v = np.clip(v, ylo, yhi)                 # safeguard: closure cannot exceed the physical training range
        n3 = len(o3)
        kap = v[:n3] + 1j * v[n3:]
        return {k: kap[i] for i, k in enumerate(o3)}

    plain_full = integrate(eng, m0, ts, None)
    learned_full = integrate(eng, m0, ts, learned_closure)
    plain = np.array([obs(m, eng, nkey, zk, pmk) for m in plain_full])
    learned = np.array([obs(m, eng, nkey, zk, pmk) for m in learned_full])
    # align lengths (a diverging run returns fewer steps)
    L = min(len(plain), len(learned), len(exact))
    plain, learned, exact = plain[:L], learned[:L], exact[:L]
    labels = ["<a^dag a>", "mean<sz>", "mean Re<s+ s->"]
    stable = (len(learned_full) == NT) and np.all(np.isfinite(learned)) and np.max(np.abs(learned)) < 1e3
    print(f"\nHELD-OUT seed 7, sigma=0.15  (closure never saw this realization)")
    print(f"closure stable in loop: {stable}   (learned steps completed: {len(learned_full)}/{NT})")
    print(f"{'observable':16s} {'plain-cumulant err':>20s} {'LEARNED-closure err':>22s} {'improvement':>13s}")
    for c in range(3):
        ep = np.max(np.abs(plain[:, c] - exact[:, c]))
        el = np.max(np.abs(learned[:, c] - exact[:, c]))
        print(f"{labels[c]:16s} {ep:20.4f} {el:22.4f} {f'{ep/max(el,1e-9):.1f}x':>13s}")
    # windowed error: the learned closure tracks well before any late-time instability
    print("\nwindowed max error (learned closure is accurate until it destabilizes):")
    for Tw in [15, 20, 25, 30, 40]:
        w = ts[:L] <= Tw
        ep = np.max(np.abs(plain[w] - exact[w]))
        el = np.max(np.abs(learned[w] - exact[w]))
        tag = "  learned BEATS plain" if el < ep else "  learned worse"
        print(f"   t<={Tw:2d}: plain max err={ep:.4f}   learned max err={el:.4f}{tag}")
    np.savez_compressed("results/loop_traj.npz", ts=ts[:L], exact=exact, plain=plain, learned=learned)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(13, 4))
        for c in range(3):
            ax[c].plot(ts, exact[:, c], "k-", lw=2, label="exact")
            ax[c].plot(ts, plain[:, c], "r--", lw=1.2, label="plain cumulant")
            ax[c].plot(ts, learned[:, c], "b-", lw=1.5, label="learned closure")
            ax[c].set_title(labels[c]); ax[c].set_xlabel("t"); ax[c].grid(alpha=.3); ax[c].legend()
        fig.suptitle("Learned closure in the loop, held-out disorder (M=3)")
        fig.tight_layout(); fig.savefig("results/learned_loop.png", dpi=130)
        print("\nsaved results/learned_loop.png")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
