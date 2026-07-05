"""
Run the learned closure in the closed loop at LARGE molecule number M, using the validated ScalableEngine
(compiled/vectorized EOM) -- the flagship: the closure runs where an exact Lindblad solve is impossible.

Every component is independently validated:
  - ScalableEngine.build_C == dense C to ~1e-14 at M=2..5 (validate_scalable.py)
  - deriv_compiled == build_C @ m_full to ~1e-15 (validate_compiled.py)
  - the vectorized closure features == the reference per-key features exactly (check_closure_vec.py)
  - the mcsolve reference == exact mesolve at M=5 within Monte-Carlo error (mc_ref.py)
  - the closure was trained ONLY on M=2,3 (held-out seeds): every test M is unseen (no leakage)

Three results (disordered emitters, seed 7, held out):
  (1) M=5 self-consistency: scalable+closure loop vs EXACT mesolve  (dense still exists at M=5)
  (2) large M: plain cumulant AND learned closure vs EXACT mcsolve at M=8,10  (no dense solve feasible)
  (3) very large M: closure runs stably at M=12,16,20 where NO exact reference exists
"""
import time
import numpy as np
import qutip as qt
from scipy.integrate import solve_ivp

from scalable_engine import ScalableEngine
from local_closure import LocalClosure, feature_plan, extract
from exact_tc import build_ops, build_H, collapse_ops, draw_disorder
from mc_ref import mc_reference

WC, GCOLL, KAPPA, GDOWN, GPHI = 1.0, 0.8, 0.1, 0.02, 0.02
TMAX, NT = 40.0, 200
NC = 12
TEST_SEED, TEST_SIGMA = 7, 0.15


def m0_superradiant(eng):
    """Product-state moments: vacuum cavity, every emitter excited (<sz_i>=+1). Nonzero only for p=q=0 with
    every emitter factor sigma^z or identity."""
    m = np.zeros(len(eng.track), complex)
    for i, (p, q, es) in enumerate(eng.track):
        if p == 0 and q == 0 and all(e in (0, 3) for e in es):
            m[i] = 1.0
    return m


class ClosureApply:
    """Predict kappa3 for exactly the order-3 monomials the compiled EOM needs (eng.o3_keys). Feature
    extraction is VECTORIZED per cumulant type (index gathers into the tracked-moment vector), so cost is a
    few numpy ops per type per step rather than a Python loop over the O(M^3) monomials -- essential at large M.
    Identical features to local_closure.extract(): [singles(re,im)*3, pairs(re,im)*3, detunings*3, context*3]."""
    def __init__(self, lc, eng, det):
        self.lc, self.eng, self.det = lc, eng, det
        keys = eng.o3_keys
        tp = eng.tpos
        by_type = {}
        for i, k in enumerate(keys):
            sig, singles, pairs, sites = feature_plan(k)
            by_type.setdefault(sig, []).append((i, singles, pairs, sites))
        self.blocks = {}
        for t, rows in by_type.items():
            out_idx = np.array([r[0] for r in rows])
            sing = np.array([[tp[s] for s in r[1]] for r in rows])       # (nk,3)
            pair = np.array([[tp[p] for p in r[2]] for r in rows])       # (nk,3)
            dfeat = np.array([[det[s] if s >= 0 else 0.0 for s in r[3]] for r in rows])  # (nk,3), constant
            self.blocks[t] = (out_idx, sing, pair, dfeat)
        self.out_len = len(keys)

    def __call__(self, m):
        eng = self.eng
        mdict = {k: m[j] for j, k in enumerate(eng.track)}
        ctx = np.array(self.lc.context(mdict, eng.M))
        out = np.zeros(self.out_len, complex)
        for t, (out_idx, sing, pair, dfeat) in self.blocks.items():
            if t not in self.lc.nets:
                continue                                   # near-zero cumulant type -> kappa3 = 0
            s = m[sing]                                     # (nk,3) complex
            p = m[pair]                                     # (nk,3) complex
            nk = s.shape[0]
            X = np.empty((nk, 18))
            X[:, 0:6:2] = s.real; X[:, 1:6:2] = s.imag
            X[:, 6:12:2] = p.real; X[:, 7:12:2] = p.imag
            X[:, 12:15] = dfeat
            X[:, 15:18] = ctx
            V = self.lc.nets[t].predict(self.lc.scalers[t].transform(X))
            lo, hi = self.lc.range[t]
            V = np.clip(V, lo, hi)
            out[out_idx] = V[:, 0] + 1j * V[:, 1]
        return out


def run(eng, closure, tmax=TMAX, nt=NT, rtol=1e-4, atol=1e-6):
    ts = np.linspace(0, tmax, nt)
    m0 = m0_superradiant(eng)

    def rhs(t, y):
        n = len(y) // 2
        m = y[:n] + 1j * y[n:]
        kap = closure(m) if closure is not None else None
        d = eng.deriv_compiled(m, kap)
        return np.concatenate([d.real, d.imag])

    y0 = np.concatenate([m0.real, m0.imag])
    sol = solve_ivp(rhs, (ts[0], ts[-1]), y0, t_eval=ts, rtol=rtol, atol=atol, method="RK45")
    n = sol.y.shape[0] // 2
    traj = np.array([sol.y[:n, i] + 1j * sol.y[n:, i] for i in range(sol.y.shape[1])])
    return ts, traj


def observables(eng, traj):
    tp = eng.tpos
    M = eng.M
    nkey = (1, 1, tuple([0] * M))
    zkeys = [(0, 0, tuple(3 if s == i else 0 for s in range(M))) for i in range(M)]
    zcols = [tp[k] for k in zkeys]
    n = np.array([row[tp[nkey]].real for row in traj])
    z = np.array([np.mean([row[c].real for c in zcols]) for row in traj])
    return n, z


def stable_ok(traj):
    return np.isfinite(traj).all() and np.max(np.abs(traj)) < 1e2 and len(traj) == NT


def exact_dense_obs(M, Nc, w, g):
    ts = np.linspace(0, TMAX, NT)
    ops = build_ops(M, Nc)
    H = build_H(ops, WC, w, g)
    cops = collapse_ops(ops, kappa=KAPPA, gdown=GDOWN, gphi=GPHI)
    psi0 = qt.tensor(qt.basis(Nc, 0), *[qt.basis(2, 0) for _ in range(M)])
    n = ops["a"].dag() * ops["a"]
    zmean = sum(ops["sz"]) / M
    e = qt.mesolve(H, psi0, ts, c_ops=cops, e_ops=[n, zmean], options={"progress_bar": False}).expect
    return np.array(e[0]).real, np.array(e[1]).real


def maxerr(n, z, ne, ze):
    L = min(len(n), len(ne))
    return max(float(np.max(np.abs(n[:L] - ne[:L]))), float(np.max(np.abs(z[:L] - ze[:L]))))


def large_ref(M, Nc, w, g, ntraj=200):
    """Exact large-M reference by quantum trajectories (mcsolve). mcsolve evolves cheap pure states of dim
    Nc*2^M; a full density-matrix mesolve (rho ~ (Nc*2^M)^2) is already impractical for the stiff superradiant
    burst by M~7. Returns (n, z, MC_error). Same exact_tc model as the validated dense engine (no convention
    risk; mcsolve validated against exact mesolve at M=5)."""
    _, n, z, mcerr = mc_reference(M, Nc, w, g, ntraj=ntraj)
    return n, z, mcerr


def train_closure():
    lc = LocalClosure(M=3)
    data = np.load("results/transfer_train.npz", allow_pickle=True)["data"].item()
    lc.fit(data)
    return lc


def build(M, sigma, seed):
    w = draw_disorder(M, WC, sigma, seed, "gauss") if sigma > 0 else np.full(M, WC)
    g = np.full(M, GCOLL / np.sqrt(M))
    eng = ScalableEngine(M, NC, WC, w, g, kappa=KAPPA, gdown=GDOWN, gphi=GPHI, Fsafe=6).compile()
    return eng, w, g


if __name__ == "__main__":
    lc = train_closure()
    dom = [t for t in lc.types if t in lc.nets and lc.nets[t].best_validation_score_ > 0.8]
    print(f"closure trained on M=2,3; dominant types (val>0.8): {len(dom)}\n")

    # (1) M=5 self-consistency vs exact
    print("(1) M=5 self-consistency: scalable+closure loop vs EXACT mesolve (disordered, seed 7)")
    eng, w, g = build(5, TEST_SIGMA, TEST_SEED)
    det = w - WC
    _, plain = run(eng, None)
    _, learned = run(eng, ClosureApply(lc, eng, det))
    ne, ze = exact_dense_obs(5, NC, w, g)
    ep = maxerr(*observables(eng, plain), ne, ze)
    el = maxerr(*observables(eng, learned), ne, ze)
    print(f"    plain err={ep:.4f}  learned err={el:.4f}  improvement={ep/max(el,1e-9):.1f}x\n")

    # (2) large M vs EXACT reference (dense mesolve M<=8, mcsolve above), DISORDERED -- extends transfer past M=5
    print("(2) large M: plain cumulant & learned closure vs EXACT reference (disordered, seed 7)")
    print(f"    {'M':>3s} {'tracked':>8s} {'plain err':>10s} {'learned err':>12s} {'refErr':>7s} {'winner':>8s} {'sec':>5s}")
    for M in [8, 10]:
        t0 = time.time()
        eng, w, g = build(M, TEST_SIGMA, TEST_SEED)
        det = w - WC
        _, plain = run(eng, None)
        _, learned = run(eng, ClosureApply(lc, eng, det))
        try:
            nref, zref, referr = large_ref(M, NC, w, g)
            ep = maxerr(*observables(eng, plain), nref, zref) if stable_ok(plain) else float("inf")
            el = maxerr(*observables(eng, learned), nref, zref) if stable_ok(learned) else float("inf")
            win = "learned" if el < ep else "plain"
            print(f"    {M:>3d} {len(eng.track):>8d} {ep:>10.4f} {el:>12.4f} {referr:>7.3f} {win:>8s} {time.time()-t0:>5.0f}")
        except Exception as ex:
            print(f"    {M:>3d} {len(eng.track):>8d}   reference failed: {type(ex).__name__} (learned stable={stable_ok(learned)})")

    # (3) very large M: stability where no exact reference exists at all
    print("\n(3) very large M (sigma=0.15): closure stability where NO exact solve is possible")
    print(f"    {'M':>3s} {'tracked':>8s} {'stable':>7s} {'n_peak':>7s} {'z_end':>7s} {'sec':>5s}")
    for M in [12, 16, 20]:
        t0 = time.time()
        eng, w, g = build(M, TEST_SIGMA, TEST_SEED)
        det = w - WC
        _, learned = run(eng, ClosureApply(lc, eng, det))
        nl, zl = observables(eng, learned)
        print(f"    {M:>3d} {len(eng.track):>8d} {str(stable_ok(learned)):>7s} {nl.max():>7.3f} "
              f"{zl[-1]:>7.3f} {time.time()-t0:>5.0f}")
