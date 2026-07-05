"""
T3 (kappa4=0 third-order truncation) closed-loop evaluation at M=3,4 (pre-registered cap). Answers the
ladder's "why not just track third order?" rung with measurements: stability, conditional NRMSE, wall
time, and tracked-moment count, on deep-corner and normal cells, vs exact mesolve references. Order-2
plain cumulant and the V7 branch run at the same cells for context. Gate < 1e-8 enforced per engine.
Default seeds 29/31 (dev); --seeds 32,33,34 for the final table.
"""
import sys
import time
import numpy as np
import qutip as qt
from scipy.integrate import solve_ivp

from t3_truncation import T3Engine
from scalable_engine import ScalableEngine
from local_run import observables, stable_ok
from pilot_eval import m0_product, exact_ref, nrmse, WC
from v7 import run_v7
from exact_tc import draw_disorder

SEEDS = [29, 31]
if "--seeds" in sys.argv:
    SEEDS = [int(s) for s in sys.argv[sys.argv.index("--seeds") + 1].split(",")]
GC, SG = 0.7, 0.15
NCS = {3: 10, 4: 11}
CELLS = [("deep corner", 0.030, 0.020), ("normal", 0.15, 0.02)]


def t3_m0(eng):
    m = np.zeros(len(eng.track), complex)
    for i, (p, q, es) in enumerate(eng.track):
        if p == 0 and q == 0 and all(e in (0, 3) for e in es):
            m[i] = 1.0
    return m


def t3_obs(eng, traj):
    tp = {k: i for i, k in enumerate(eng.track)}
    M = eng.M
    n = np.array([row[tp[(1, 1, tuple([0] * M))]].real for row in traj])
    z = np.array([np.mean([row[tp[(0, 0, tuple(3 if s == i else 0 for s in range(M)))]].real
                           for i in range(M)]) for row in traj])
    return n, z


def run_t3(eng):
    ts = np.linspace(0, 40.0, 200)
    m0 = t3_m0(eng)

    def rhs(t, y):
        n = len(y) // 2
        d = eng.deriv(y[:n] + 1j * y[n:])
        return np.concatenate([d.real, d.imag])

    t0 = time.time()
    sol = solve_ivp(rhs, (0, 40.0), np.concatenate([m0.real, m0.imag]), t_eval=ts,
                    rtol=1e-4, atol=1e-6, method="RK45")
    n = sol.y.shape[0] // 2
    traj = np.array([sol.y[:n, i] + 1j * sol.y[n:, i] for i in range(sol.y.shape[1])])
    return traj, time.time() - t0


if __name__ == "__main__":
    print(f"T3 closed-loop eval, M={list(NCS)}, seeds {SEEDS}, cells {CELLS}\n")
    for M, Nc in NCS.items():
        for (tag, ka, gp) in CELLS:
            for sd in SEEDS:
                w = draw_disorder(M, WC, SG, sd, "gauss")
                g = np.full(M, GC / np.sqrt(M))
                t0 = time.time()
                eng3 = T3Engine(M, Nc, WC, w, g, kappa=ka, gdown=0.02, gphi=gp, Fsafe=6)
                build_s = time.time() - t0
                psi0 = qt.tensor(qt.basis(Nc, 0), *[qt.basis(2, 0) for _ in range(M)])
                rho = qt.mesolve(eng3.H, psi0, np.linspace(0, 4, 2), c_ops=eng3.cops,
                                 options={"progress_bar": False}).states[-1]
                gate = eng3.gate_exact(rho)
                assert gate < 1e-8, f"T3 gate FAILED ({gate:.1e}) at M={M} {tag} -- rollout not trusted"
                ne, ze, _ = exact_ref(M, w, g, ka, gp)   # exact_ref builds its own ops at NC=12 cavity
                traj3, dt3 = run_t3(eng3)
                ok3 = np.isfinite(traj3).all() and np.max(np.abs(traj3)) < 1e2 and len(traj3) == 200
                n3, z3 = t3_obs(eng3, traj3)
                v3v = nrmse(n3, ne) if ok3 else np.inf
                # context: order-2 plain + V7 at the same cell
                eng2 = ScalableEngine(M, 12, WC, w, g, kappa=ka, gdown=0.02, gphi=gp, Fsafe=6).compile()
                det = w - WC
                m0 = m0_product(eng2, 1.0)
                from pilot_eval import run_from
                _, tp2 = run_from(eng2, m0, None)
                p2 = nrmse(observables(eng2, tp2)[0], ne) if stable_ok(tp2) else np.inf
                br, t7 = run_v7(eng2, m0, det, (GC, ka, gp))
                v7v = nrmse(observables(eng2, t7)[0], ne) if stable_ok(t7) else np.inf
                print(f"  M={M} {tag:>11s} ka={ka:.3f} gp={gp:.3f} s{sd}: gate={gate:.1e} "
                      f"T3={'X' if not ok3 else f'{v3v:.3f}'} (tracked {len(eng3.track)}, build {build_s:.0f}s, "
                      f"roll {dt3:.1f}s) | plain2={'X' if not np.isfinite(p2) else f'{p2:.3f}'} "
                      f"| V7[{br}]={'X' if not np.isfinite(v7v) else f'{v7v:.3f}'}", flush=True)
    print("\n(scaling context: order-2 tracked = 60/96/141 at M=3/4/5; T3 tracked grows ~O(M^3-4) and the")
    print(" dense T3 build is already minutes at M=4 -- the 'just track third order' cost curve.)")
