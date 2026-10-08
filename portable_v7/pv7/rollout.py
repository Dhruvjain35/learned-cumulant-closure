"""Cell construction, rollouts and observables, mirroring src/evaluation/pilot_eval.py (build_cell, m0_product,
run_from), src/closures/closure_v6.py (run_v6), src/closures/v7.py (run_v7), src/evaluation/local_run.py
(observables, stable_ok) and src/evaluation/burst_pred.py (exact_ref_big, burst_obs). Same constants, same solver
settings."""
import numpy as np
import qutip as qt
from scipy.integrate import solve_ivp

from .engine.exact_tc import build_ops, build_H, collapse_ops, draw_disorder
from .engine.scalable_engine import ScalableEngine
from .features import context_v3, lambdas
from .closure import ClosureApply
from .model import Branch
from .router import route

WC, GDOWN, NC, FSAFE = 1.0, 0.02, 12, 6
TMAX, NT = 40.0, 200
NC_REF = 16
_BRANCHES = {}


def branch(name):
    if name not in _BRANCHES:
        _BRANCHES[name] = Branch(name)
    return _BRANCHES[name]


def build_cell(M, gc, ka, gp, sg, seed):
    w = draw_disorder(M, WC, sg, seed, "gauss")
    g = np.full(M, gc / np.sqrt(M))
    eng = ScalableEngine(M, NC, WC, w, g, kappa=ka, gdown=GDOWN, gphi=gp, Fsafe=FSAFE).compile()
    return eng, w, g


def m0_product(eng, f=1.0):
    m = np.zeros(len(eng.track), complex)
    for i, (p, q, es) in enumerate(eng.track):
        if p == 0 and q == 0 and all(e in (0, 3) for e in es):
            m[i] = (2.0 * f - 1.0) ** sum(1 for e in es if e == 3)
    return m


def run_from(eng, m0, closure=None, rtol=1e-4, atol=1e-6):
    ts = np.linspace(0, TMAX, NT)

    def rhs(t, y):
        n = len(y) // 2
        mm = y[:n] + 1j * y[n:]
        kap = closure(mm) if closure is not None else None
        d = eng.deriv_compiled(mm, kap)
        return np.concatenate([d.real, d.imag])

    sol = solve_ivp(rhs, (ts[0], ts[-1]), np.concatenate([m0.real, m0.imag]),
                    t_eval=ts, rtol=rtol, atol=atol, method="RK45")
    n = sol.y.shape[0] // 2
    return ts, np.array([sol.y[:n, i] + 1j * sol.y[n:, i] for i in range(sol.y.shape[1])])


def run_v6(eng, m0, clo, lam, tmax=40.0, nt=200, rtol=1e-4, atol=1e-6):
    ts = np.linspace(0, tmax, nt)
    nm = len(m0)
    h0 = np.array(context_v3({k: m0[i] for i, k in enumerate(eng.track)}, eng.M))

    def rhs(t, y):
        m = y[:nm] + 1j * y[nm:2 * nm]
        h = y[2 * nm:]
        clo.current_h = h
        kap = clo(m)
        dm = eng.deriv_compiled(m, kap)
        x = np.array(context_v3({k: m[i] for i, k in enumerate(eng.track)}, eng.M))
        return np.concatenate([dm.real, dm.imag, lam * (x - h)])

    sol = solve_ivp(rhs, (ts[0], ts[-1]), np.concatenate([m0.real, m0.imag, h0]),
                    t_eval=ts, rtol=rtol, atol=atol, method="RK45")
    traj = np.array([sol.y[:nm, i] + 1j * sol.y[nm:2 * nm, i] for i in range(sol.y.shape[1])])
    return ts, traj, sol.y[2 * nm:, :].T


def run_v7(eng, m0, det, regime):
    """regime = (g_c, kappa, gamma_phi). Returns (branch_name, trajectory)."""
    gc, ka, gp = regime
    b = route(ka, gp)
    if b == "v6":
        clo = ClosureApply(branch("v6"), eng, det, regime, memory=True)
        _, traj, _ = run_v6(eng, m0, clo, lambdas(ka, gp))
    else:
        clo = ClosureApply(branch("v4"), eng, det, regime, memory=False)
        _, traj = run_from(eng, m0, clo)
    return b, traj


def observables(eng, traj):
    tp = eng.tpos
    M = eng.M
    nkey = (1, 1, tuple([0] * M))
    zcols = [tp[(0, 0, tuple(3 if s == i else 0 for s in range(M)))] for i in range(M)]
    n = np.array([row[tp[nkey]].real for row in traj])
    z = np.array([np.mean([row[c].real for c in zcols]) for row in traj])
    return n, z


def stable_ok(traj):
    return np.isfinite(traj).all() and np.max(np.abs(traj)) < 1e2 and len(traj) == NT


def physicality(eng, traj, ts, tmax, tol=1e-6):
    tp = eng.tpos
    nkey = (1, 1, tuple([0] * eng.M))
    zcols = [tp[(0, 0, tuple(3 if s == i else 0 for s in range(eng.M)))] for i in range(eng.M)]
    sel = ts <= tmax + 1e-12
    sz = np.array([[row[c].real for c in zcols] for row in traj[sel]])
    n = np.array([row[tp[nkey]].real for row in traj[sel]])
    return bool(np.max(np.abs(sz)) - 1.0 <= tol and np.min(n) >= -tol)


def exact_ref(M, w, g, ka, gp, Nc=NC_REF):
    ops = build_ops(M, Nc)
    H = build_H(ops, WC, w, g)
    cops = collapse_ops(ops, kappa=ka, gdown=GDOWN, gphi=gp)
    psi0 = qt.tensor(qt.basis(Nc, 0), *[qt.basis(2, 0) for _ in range(M)])
    n = ops["a"].dag() * ops["a"]
    zmean = sum(ops["sz"]) / M
    ts = np.linspace(0, TMAX, NT)
    e = qt.mesolve(H, psi0, ts, c_ops=cops, e_ops=[n, zmean], options={"progress_bar": False}).expect
    ne = np.array(e[0]).real
    return ts, ne, np.array(e[1]).real, float(ne.max()) / Nc


def burst_obs(ts, n, kappa):
    n = np.asarray(n).real
    ipk = int(np.argmax(n))
    return dict(Fpeak=kappa * float(n[ipk]), tpeak=float(ts[ipk]),
                Itot=kappa * float(np.trapezoid(n, ts)), npeak=float(n[ipk]))
