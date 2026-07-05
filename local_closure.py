"""
Local-feature, permutation-equivariant closure for the FULL order-3 cumulant set, used in the loop.

For each order-3 monomial we predict its cumulant kappa3 from ONLY local information: the moments among
its own fundamental factors (the singles <f_i> and pairs <f_i f_j> that appear in its cumulant
factorization), the detunings of its emitter sites, and a little global context (photon number, mean
inversion, mean coherence). Monomials are canonicalized by factor-type so the same network serves every
site assignment and every M -> permutation-equivariant and transferable across emitter number by design.

We train one small network per canonical type signature, then integrate the cumulant EOM with this closure
supplied live, on a held-out disorder realization, and (transfer) at a larger M than trained.
"""
import os
import numpy as np
import qutip as qt
from scipy.integrate import solve_ivp
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from cumulant import CumulantEngine, factors_of, pair_key, mono_order
from exact_tc import draw_disorder

WC, GCOLL, KAPPA, GDOWN, GPHI = 1.0, 0.8, 0.1, 0.02, 0.02
NC, FSAFE = 12, 6
TMAX, NT = 40.0, 200
OPLAB = {(1, 0): "ad", (0, 1): "a"}          # cavity factor labels; emitter via emitter tuple


def factor_info(fk):
    p, q, es = fk
    if p == 1 and q == 0:
        return "ad", -1
    if p == 0 and q == 1:
        return "a", -1
    for i, e in enumerate(es):
        if e != 0:
            return {1: "sp", 2: "sm", 3: "sz"}[e], i
    raise ValueError(fk)


TYPE_ORDER = {"ad": 0, "a": 1, "sp": 2, "sm": 3, "sz": 4}


def canonical(mono_key):
    """Return (type_signature, ordered factor keys, ordered (label,site)) with emitters sorted by type."""
    facs = factors_of(mono_key)
    info = [factor_info(f) for f in facs]
    order = sorted(range(3), key=lambda k: (TYPE_ORDER[info[k][0]], info[k][1]))
    fk = [facs[k] for k in order]
    lab = [info[k] for k in order]
    sig = tuple(l[0] for l in lab)
    return sig, fk, lab


def feature_plan(mono_key):
    """Precompute the tracked-moment keys needed for this monomial's local features."""
    sig, fk, lab = canonical(mono_key)
    singles = fk                                   # 3 order-1 keys
    pairs = [pair_key(fk[0], fk[1]), pair_key(fk[0], fk[2]), pair_key(fk[1], fk[2])]
    sites = [l[1] for l in lab]
    return sig, singles, pairs, sites


def extract(mdict, plan, det, ctx):
    sig, singles, pairs, sites = plan
    f = []
    for k in singles:
        z = mdict[k]; f += [z.real, z.imag]
    for k in pairs:
        z = mdict[k]; f += [z.real, z.imag]
    f += [det[s] if s >= 0 else 0.0 for s in sites]
    f += ctx
    return f


class LocalClosure:
    def __init__(self, M):
        self.M = M
        # build a template engine just to enumerate order-3 monomials and feature plans (params irrelevant)
        eng = CumulantEngine(M, NC, WC, np.full(M, WC), np.full(M, GCOLL / np.sqrt(M)),
                             kappa=KAPPA, gdown=GDOWN, gphi=GPHI, Fsafe=FSAFE)
        self.order3 = eng.order3
        self.plans = {k: feature_plan(k) for k in self.order3}
        self.types = sorted(set(p[0] for p in self.plans.values()))
        self.nets = {}
        self.scalers = {}

    def context(self, mdict, M):
        nc = mdict[(1, 1, tuple([0] * M))].real
        zs = [mdict[(0, 0, tuple(3 if s == i else 0 for s in range(M)))].real for i in range(M)]
        coh = [abs(mdict[(0, 0, tuple({i: 1, j: 2}.get(s, 0) for s in range(M)))])
               for i in range(M) for j in range(M) if i != j]
        return [nc, float(np.mean(zs)), float(np.mean(coh)) if coh else 0.0]

    def dataset(self, engines_states_dets):
        """engines_states_dets: list of (eng, states, det). Returns per-type (X, Y)."""
        Xd = {t: [] for t in self.types}
        Yd = {t: [] for t in self.types}
        for eng, states, det in engines_states_dets:
            plans = {k: feature_plan(k) for k in eng.order3}
            for r in states:
                mfull = eng.moments_from_rho(r)
                mdict = {k: mfull[eng.idx[k]] for k in eng.full}
                mt = mfull[eng.tidx]
                base = eng._full_from_tracked(mt, {})
                ctx = self.context(mdict, eng.M)
                for k in eng.order3:
                    plan = plans[k]
                    Xd[plan[0]].append(extract(mdict, plan, det, ctx))
                    kap = mfull[eng.idx[k]] - base[eng.idx[k]]
                    Yd[plan[0]].append([kap.real, kap.imag])
        return {t: (np.array(Xd[t]), np.array(Yd[t])) for t in self.types}, Xd, Yd

    def fit(self, data):
        self.range = {}
        for t, (X, Y) in data.items():
            if len(X) < 20:
                continue
            sc = StandardScaler().fit(X)
            net = MLPRegressor(hidden_layer_sizes=(128, 128), alpha=1e-5, max_iter=1500,
                               early_stopping=True, n_iter_no_change=25, random_state=0)
            net.fit(sc.transform(X), Y)
            self.scalers[t], self.nets[t] = sc, net
            self.range[t] = (Y.min(0), Y.max(0))

    def closure_fn(self, eng, det):
        plans = {k: feature_plan(k) for k in eng.order3}
        by_type = {}
        for k in eng.order3:
            by_type.setdefault(plans[k][0], []).append(k)

        def fn(mtrack):
            mdict = {k: mtrack[i] for i, k in enumerate(eng.track)}
            ctx = self.context(mdict, eng.M)
            out = {}
            for t, keys in by_type.items():          # one batched predict per cumulant type
                if t not in self.nets:
                    for k in keys:
                        out[k] = 0.0
                    continue
                X = np.array([extract(mdict, plans[k], det, ctx) for k in keys])
                V = self.nets[t].predict(self.scalers[t].transform(X))
                lo, hi = self.range[t]
                V = np.clip(V, lo, hi)               # physical safeguard
                for i, k in enumerate(keys):
                    out[k] = V[i, 0] + 1j * V[i, 1]
            return out
        return fn


def make_engine(M, sigma, seed):
    w = draw_disorder(M, WC, sigma, seed, "gauss")
    g = np.full(M, GCOLL / np.sqrt(M))
    eng = CumulantEngine(M, NC, WC, w, g, kappa=KAPPA, gdown=GDOWN, gphi=GPHI, Fsafe=FSAFE)
    return eng, (w - WC)


def exact_states(eng):
    ts = np.linspace(0, TMAX, NT)
    psi0 = qt.tensor(qt.basis(NC, 0), *[qt.basis(2, 0) for _ in range(eng.M)])
    return ts, qt.mesolve(eng.H, psi0, ts, c_ops=eng.cops, options={"progress_bar": False}).states, psi0


def obs(mtrack, eng):
    ti = {k: a for a, k in enumerate(eng.track)}
    M = eng.M
    n = mtrack[ti[(1, 1, tuple([0] * M))]].real
    z = np.mean([mtrack[ti[(0, 0, tuple(3 if s == i else 0 for s in range(M)))]].real for i in range(M)])
    pm = np.mean([mtrack[ti[(0, 0, tuple({i: 1, j: 2}.get(s, 0) for s in range(M)))]].real
                  for i in range(M) for j in range(M) if i != j])
    return n, z, pm


def pack(m): return np.concatenate([m.real, m.imag])
def unpack(y): n = len(y) // 2; return y[:n] + 1j * y[n:]


def run_loop(eng, m0, ts, closure=None):
    o3 = eng.order3
    def rhs(t, y):
        m = unpack(y)
        kap = None
        if closure:
            k3 = closure(m)
            kap = np.fromiter((k3.get(k, 0.0) for k in o3), complex, count=len(o3))
        return pack(eng.deriv_fast(m, kap))
    sol = solve_ivp(rhs, (ts[0], ts[-1]), pack(m0), t_eval=ts, rtol=1e-4, atol=1e-6, method="RK45")
    return np.array([unpack(sol.y[:, i]) for i in range(sol.y.shape[1])])


def windowed(ts, exact, plain, learned, L, tag):
    print(f"\n[{tag}] windowed max error (over 3 observables):")
    for Tw in [15, 20, 25, 30, 40]:
        w = ts[:L] <= Tw
        ep = np.max(np.abs(plain[w] - exact[w])); el = np.max(np.abs(learned[w] - exact[w]))
        print(f"   t<={Tw:2d}: plain={ep:.4f}  learned(local)={el:.4f}  {'BEATS' if el<ep else 'worse'}")


def main():
    lc = LocalClosure(M=3)
    print(f"order-3 monomials: {len(lc.order3)}   canonical types: {lc.types}")

    cache = "results/local_train.npz"
    train_specs = [(3, s, sd) for s in [0.05, 0.10, 0.15, 0.20, 0.30] for sd in [0, 1, 2, 3, 4, 5]]
    if os.path.exists(cache):
        print("loading cached per-type dataset...")
        d = np.load(cache, allow_pickle=True); data = d["data"].item()
    else:
        print("generating training data (exact trajectories)...")
        esd = []
        for (M, sg, sd) in train_specs:
            eng, det = make_engine(M, sg, sd)
            _, states, _ = exact_states(eng)
            esd.append((eng, states, det))
        data, _, _ = lc.dataset(esd)
        np.savez_compressed(cache, data=np.array(data, dtype=object))
    lc.fit(data)
    for t in lc.types:
        n = len(data[t][0]) if t in data else 0
        vs = lc.nets[t].best_validation_score_ if t in lc.nets else float("nan")
        print(f"   type {str(t):18s}: {n:6d} samples  val_score={vs:.3f}")

    # ---- held-out M=3 loop ----
    eng, det = make_engine(3, 0.15, 7)
    ts, states, psi0 = exact_states(eng)
    exact = np.array([obs(eng.moments_from_rho(r)[eng.tidx], eng) for r in states])
    m0 = eng.moments_from_rho(psi0)[eng.tidx]
    plain = np.array([obs(m, eng) for m in run_loop(eng, m0, ts, None)])
    learned_traj = run_loop(eng, m0, ts, lc.closure_fn(eng, det))
    learned = np.array([obs(m, eng) for m in learned_traj])
    L = min(len(exact), len(plain), len(learned))
    print(f"\nHELD-OUT M=3 seed 7: learned steps {len(learned_traj)}/{NT}")
    windowed(ts, exact[:L], plain[:L], learned[:L], L, "M=3 held-out (local closure)")

    # ---- transfer: run the loop at M=4 with the M=3-trained closure ----
    eng4, det4 = make_engine(4, 0.15, 3)
    ts4, states4, psi04 = exact_states(eng4)
    exact4 = np.array([obs(eng4.moments_from_rho(r)[eng4.tidx], eng4) for r in states4])
    m04 = eng4.moments_from_rho(psi04)[eng4.tidx]
    plain4 = np.array([obs(m, eng4) for m in run_loop(eng4, m04, ts4, None)])
    learned4_traj = run_loop(eng4, m04, ts4, lc.closure_fn(eng4, det4))
    learned4 = np.array([obs(m, eng4) for m in learned4_traj])
    L4 = min(len(exact4), len(plain4), len(learned4))
    print(f"\nTRANSFER to M=4 (closure trained only at M=3): learned steps {len(learned4_traj)}/{NT}")
    windowed(ts4, exact4[:L4], plain4[:L4], learned4[:L4], L4, "M=4 transfer (local closure trained at M=3)")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 3, figsize=(13, 7.5))
        labels = ["<a^dag a>", "mean<sz>", "mean Re<s+ s->"]
        for c in range(3):
            ax[0, c].plot(ts[:L], exact[:L, c], "k-", lw=2, label="exact")
            ax[0, c].plot(ts[:L], plain[:L, c], "r--", lw=1.1, label="plain")
            ax[0, c].plot(ts[:L], learned[:L, c], "b-", lw=1.4, label="local closure")
            ax[0, c].set_title("M=3 held-out: " + labels[c]); ax[0, c].grid(alpha=.3); ax[0, c].legend(fontsize=8)
            ax[1, c].plot(ts4[:L4], exact4[:L4, c], "k-", lw=2, label="exact")
            ax[1, c].plot(ts4[:L4], plain4[:L4, c], "r--", lw=1.1, label="plain")
            ax[1, c].plot(ts4[:L4], learned4[:L4, c], "b-", lw=1.4, label="local closure")
            ax[1, c].set_title("M=4 transfer: " + labels[c]); ax[1, c].set_xlabel("t"); ax[1, c].grid(alpha=.3); ax[1, c].legend(fontsize=8)
        fig.tight_layout(); fig.savefig("results/local_closure.png", dpi=130)
        print("\nsaved results/local_closure.png")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
