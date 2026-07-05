"""
EXPORT of already-computed frozen trajectories (deterministic; NO model change, NO new science, NO retuning).

Runs the frozen engines ONCE at M=3, long-coherence (POCKET) regime, disorder draw 50, and serializes the
cavity-occupation trace n(t) = <a^dag a>(t) for: exact / 2nd-order / kappa4=0 (T3) / kappa5=0 (T4, integrated
only up to the divergence guard) / V7. Output feeds the pure-plotting script (which reads saved data only).

The models here are frozen and the parameters + disorder draw are fixed, so this re-execution reproduces the
same trajectories that the burst tests already computed; it only serializes them to disk.

Run from the repo root:  python run.py scripts/export_baseline_burst_trace.py
"""
import numpy as np, pickle
from scipy.integrate import solve_ivp

from pilot_eval import build_cell, m0_product, run_from, WC
from v7 import run_v7
from local_run import observables
from exact_tc import draw_disorder
from t3_truncation import T3Engine
from t4_truncation import T4Engine
from burst_pred import exact_ref_big, NC_REF

GC, KA, GP, GDOWN, SG = 0.70, 0.03, 0.02, 0.02, 0.15
M, SEED = 3, 50
TMAX, NT = 40.0, 200
GUARD = 1.0e4
OUT = "results/baseline_burst_trace_M3_pocket_draw50.npz"


def roll_trunc(eng):
    """Roll a truncation engine (T3/T4) from the inverted state, stopping at the runaway/divergence guard.
    Returns (t, n(t), t_stop, guard_triggered)."""
    m0 = np.zeros(len(eng.track), complex)
    for i, (p, q, es) in enumerate(eng.track):
        if p == 0 and q == 0 and all(e in (0, 3) for e in es):
            m0[i] = 1.0

    def rhs(t, y):
        n = len(y) // 2
        d = eng.deriv(y[:n] + 1j * y[n:])
        return np.concatenate([d.real, d.imag])

    def blow(t, y):
        return GUARD - float(np.max(np.abs(y)))
    blow.terminal = True; blow.direction = -1
    sol = solve_ivp(rhs, (0, TMAX), np.concatenate([m0.real, m0.imag]),
                    t_eval=np.linspace(0, TMAX, NT), rtol=1e-5, atol=1e-7,
                    method="LSODA", events=blow, max_step=0.5)
    ni = eng.track.index((1, 1, tuple(0 for _ in range(eng.M))))
    n = (sol.y[ni] + 1j * sol.y[ni + len(eng.track)]).real
    t_stop = float(sol.t[-1])
    return sol.t, n, t_stop, (t_stop < TMAX - 1e-6)


if __name__ == "__main__":
    w = draw_disorder(M, WC, SG, SEED, "gauss")
    g = np.full(M, GC / np.sqrt(M))
    eng, _, _ = build_cell(M, GC, KA, GP, SG, SEED)
    det = w - WC
    m0 = m0_product(eng, 1.0)
    ts = np.linspace(0, TMAX, NT)

    ts_ex, n_ex, _, _ = exact_ref_big(M, w, g, KA, GP, NC_REF)             # exact
    _, tp = run_from(eng, m0, None); n_plain = observables(eng, tp)[0]     # 2nd order
    branch, t7 = run_v7(eng, m0, det, (GC, KA, GP)); n_v7 = observables(eng, t7)[0]   # V7
    e3 = T3Engine(M, 10, WC, w, g, kappa=KA, gdown=GDOWN, gphi=GP, Fsafe=6)
    ts3, n3, stop3, trig3 = roll_trunc(e3)                                 # kappa4=0
    e4 = T4Engine(M, 10, WC, w, g, kappa=KA, gdown=GDOWN, gphi=GP, Fsafe=6)
    ts4, n4, stop4, trig4 = roll_trunc(e4)                                 # kappa5=0 (to guard)

    meta = dict(M=M, regime="POCKET (long-coherence)", gc=GC, kappa=KA, gphi=GP, gdown=GDOWN,
                sigma=SG, seed=SEED, v7_branch=branch, guard=GUARD,
                t3_stop=stop3, t3_guard_triggered=bool(trig3),
                t4_stop=stop4, t4_guard_triggered=bool(trig4))
    np.savez_compressed(OUT, meta=meta,
                        ts_exact=ts_ex, n_exact=n_ex, ts_plain=ts, n_plain=n_plain,
                        ts_v7=ts, n_v7=n_v7, ts_t3=ts3, n_t3=n3, ts_t4=ts4, n_t4=n4)
    print(f"exported {OUT}")
    print(f"  V7 branch={branch}; T4 guard triggered={trig4} at t={stop4:.3f}; "
          f"T3 guard triggered={trig3} at t={stop3:.3f}")
