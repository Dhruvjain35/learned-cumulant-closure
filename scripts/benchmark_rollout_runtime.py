"""
FROZEN-RUNTIME BENCHMARK. Measures build vs rollout vs total wall-clock for the tested closures on the SAME M=3
cells (CONTROL and long-coherence POCKET, disorder draws 50-52) used in the fourth-order fairness test.

NO retrain, NO reroute, NO tune, NO model change. Nothing here alters V7 or any engine; it only times them.

Methods: 2nd-order (plain) | kappa4=0 (T3) | V7 (routed learned) | kappa5=0 (T4).
  - build time    = constructing everything needed BEFORE integration begins:
      * 2nd-order / V7 : the scalable engine (build_cell); V7 additionally constructs the closure applier.
                         (The frozen model pickles are loaded once, outside the timed loop.)
      * T3 / T4       : the truncation engine (T3Engine / T4Engine).
  - rollout time  = the time integration itself (run_from / run_v6 / solve_ivp).
      * for T4 the integration is timed until the runaway-growth guard triggers; the guard's simulation-time t
        is recorded separately (t4_guard_simtime).
  - total         = build + rollout.

Repeats: stable/fast methods (2nd, V7, T3) get one untimed warm-up then N_FAST timed repeats per cell.
  T4 is timed ONCE per cell (build 72-1068 s observed previously; repeated timing is impractical) -> its
  spread is taken ACROSS the 6 cells. This deviation is recorded in the report.

Run from the repo root:  python run.py scripts/benchmark_rollout_runtime.py  [--smoke]
Writes: results/runtime_baseline_comparison.csv
"""
import time, platform, pickle, sys, csv
import numpy as np
from scipy.integrate import solve_ivp

from pilot_eval import build_cell, m0_product, run_from, WC
from closure_v3 import ClosureApplyV3, ratio_transform
from closure_v6 import ClosureApplyV6, run_v6, lambdas
from v7 import route
from exact_tc import draw_disorder
from t3_truncation import T3Engine
from t4_truncation import T4Engine

REGIMES = {"CONTROL": (0.70, 0.15, 0.02), "POCKET": (0.70, 0.03, 0.02)}
DRAWS = [50, 51, 52]
GDOWN, SG, M = 0.02, 0.15, 3
TMAX, NT = 40.0, 200
NC_TRUNC = 10
GUARD = 1.0e4
N_FAST, N_T4 = 5, 1
SMOKE = "--smoke" in sys.argv
if SMOKE:
    REGIMES = {"CONTROL": (0.70, 0.15, 0.02)}; DRAWS = [50]; N_FAST = 2


def clk():
    return time.perf_counter()


def m0_trunc(eng):
    m = np.zeros(len(eng.track), complex)
    for i, (p, q, es) in enumerate(eng.track):
        if p == 0 and q == 0 and all(e in (0, 3) for e in es):
            m[i] = 1.0
    return m


def roll_trunc_timed(eng):
    """Integrate a truncation engine to t=TMAX or the runaway guard. Returns (wall_seconds, guard_simtime)."""
    m0 = m0_trunc(eng)

    def rhs(t, y):
        n = len(y) // 2
        d = eng.deriv(y[:n] + 1j * y[n:])
        return np.concatenate([d.real, d.imag])

    def blow(t, y):
        return GUARD - float(np.max(np.abs(y)))
    blow.terminal = True; blow.direction = -1
    t0 = clk()
    sol = solve_ivp(rhs, (0, TMAX), np.concatenate([m0.real, m0.imag]), t_eval=np.linspace(0, TMAX, NT),
                    rtol=1e-5, atol=1e-7, method="LSODA", events=blow, max_step=0.5)
    return clk() - t0, float(sol.t[-1])


def st(xs):
    a = np.array(xs, float)
    return (float(np.median(a)), float(a.min()), float(a.max()), len(a))


def env():
    def ver(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return "?"
    return dict(platform=platform.platform(), processor=platform.processor() or platform.machine(),
                python=sys.version.split()[0], numpy=ver("numpy"), scipy=ver("scipy"), qutip=ver("qutip"),
                tolerances="rtol=1e-5, atol=1e-7 (truncation rollouts); run_from/run_v6 defaults (2nd/V7)",
                horizon=f"t in [0,{TMAX}], {NT} points", trunc_Nc=NC_TRUNC)


if __name__ == "__main__":
    v4 = pickle.load(open("results/lc_v3dens.pkl", "rb"))
    v6 = pickle.load(open("results/lc_v6.pkl", "rb"))
    rows = []
    t_start = clk()
    E = env()
    print("ENV:", E)
    for regime, (gc, ka, gp) in REGIMES.items():
        for sd in DRAWS:
            w = draw_disorder(M, WC, SG, sd, "gauss"); g = np.full(M, gc / np.sqrt(M))
            det = w - WC
            branch = route(ka, gp)
            print(f"\n[{regime} draw {sd}]  V7 branch={branch}", flush=True)

            # ---- 2nd-order (plain): build = build_cell; rollout = run_from(None) ----
            def meas_plain():
                t0 = clk(); eng = build_cell(M, gc, ka, gp, SG, sd)[0]; b = clk() - t0
                m0 = m0_product(eng, 1.0)
                t0 = clk(); run_from(eng, m0, None); r = clk() - t0
                return b, r
            meas_plain()  # warm-up
            B, R = zip(*[meas_plain() for _ in range(N_FAST)])
            rows.append(dict(method="2nd-order", regime=regime, draw=sd, n=N_FAST, stable=True,
                             **{f"build_{k}": v for k, v in zip(("med", "min", "max", "n"), st(B))},
                             **{f"roll_{k}": v for k, v in zip(("med", "min", "max", "n"), st(R))},
                             **{f"total_{k}": v for k, v in zip(("med", "min", "max", "n"), st([b + r for b, r in zip(B, R)]))},
                             t4_guard_simtime="", note=""))
            print(f"  2nd-order  build~{np.median(B):.2f}s roll~{np.median(R):.3f}s", flush=True)

            # ---- V7: build = build_cell + applier; rollout = run_v6/run_from ----
            def meas_v7():
                t0 = clk()
                eng = build_cell(M, gc, ka, gp, SG, sd)[0]
                clo = (ClosureApplyV6(v6, eng, det, (gc, ka, gp), xform=ratio_transform) if branch == "v6"
                       else ClosureApplyV3(v4, eng, det, (gc, ka, gp), xform=ratio_transform))
                b = clk() - t0
                m0 = m0_product(eng, 1.0)
                t0 = clk()
                if branch == "v6":
                    run_v6(eng, m0, clo, lambdas(ka, gp))
                else:
                    run_from(eng, m0, clo)
                r = clk() - t0
                return b, r
            meas_v7()  # warm-up
            B, R = zip(*[meas_v7() for _ in range(N_FAST)])
            rows.append(dict(method="V7", regime=regime, draw=sd, n=N_FAST, stable=True,
                             **{f"build_{k}": v for k, v in zip(("med", "min", "max", "n"), st(B))},
                             **{f"roll_{k}": v for k, v in zip(("med", "min", "max", "n"), st(R))},
                             **{f"total_{k}": v for k, v in zip(("med", "min", "max", "n"), st([b + r for b, r in zip(B, R)]))},
                             t4_guard_simtime="", note=f"branch={branch}"))
            print(f"  V7         build~{np.median(B):.2f}s roll~{np.median(R):.3f}s", flush=True)

            # ---- T3 (kappa4=0) ----
            def meas_t3():
                t0 = clk(); e3 = T3Engine(M, NC_TRUNC, WC, w, g, kappa=ka, gdown=GDOWN, gphi=gp, Fsafe=6); b = clk() - t0
                r, tend = roll_trunc_timed(e3)
                return b, r, tend
            try:
                meas_t3()  # warm-up
                res = [meas_t3() for _ in range(N_FAST)]
                B = [x[0] for x in res]; R = [x[1] for x in res]; tend = res[-1][2]
                rows.append(dict(method="kappa4=0 (T3)", regime=regime, draw=sd, n=N_FAST, stable=(tend >= TMAX - 1e-6),
                                 **{f"build_{k}": v for k, v in zip(("med", "min", "max", "n"), st(B))},
                                 **{f"roll_{k}": v for k, v in zip(("med", "min", "max", "n"), st(R))},
                                 **{f"total_{k}": v for k, v in zip(("med", "min", "max", "n"), st([b + r for b, r in zip(B, R)]))},
                                 t4_guard_simtime=(round(tend, 3) if tend < TMAX - 1e-6 else ""),
                                 note=("guard triggered" if tend < TMAX - 1e-6 else "ran to horizon")))
                print(f"  T3         build~{np.median(B):.2f}s roll~{np.median(R):.3f}s end_t={tend:.2f}", flush=True)
            except Exception as e:
                rows.append(dict(method="kappa4=0 (T3)", regime=regime, draw=sd, n=0, stable=False, note=f"ERROR {str(e)[:50]}"))

            # ---- T4 (kappa5=0): single timed run (build+rollout too costly to repeat) ----
            try:
                t0 = clk(); e4 = T4Engine(M, NC_TRUNC, WC, w, g, kappa=ka, gdown=GDOWN, gphi=gp, Fsafe=6); b = clk() - t0
                r, tend = roll_trunc_timed(e4)
                rows.append(dict(method="kappa5=0 (T4)", regime=regime, draw=sd, n=N_T4, stable=(tend >= TMAX - 1e-6),
                                 build_med=round(b, 2), build_min=round(b, 2), build_max=round(b, 2), build_n=N_T4,
                                 roll_med=round(r, 2), roll_min=round(r, 2), roll_max=round(r, 2), roll_n=N_T4,
                                 total_med=round(b + r, 2), total_min=round(b + r, 2), total_max=round(b + r, 2), total_n=N_T4,
                                 t4_guard_simtime=round(tend, 3),
                                 note=("guard triggered (single run/cell)" if tend < TMAX - 1e-6 else "ran to horizon (single run/cell)")))
                print(f"  T4         build~{b:.1f}s roll~{r:.1f}s guard_t={tend:.2f}", flush=True)
            except Exception as e:
                rows.append(dict(method="kappa5=0 (T4)", regime=regime, draw=sd, n=0, stable=False, note=f"ERROR {str(e)[:50]}"))

    cols = ["method", "regime", "draw", "stable", "n",
            "build_med", "build_min", "build_max", "roll_med", "roll_min", "roll_max",
            "total_med", "total_min", "total_max", "t4_guard_simtime", "note"]
    with open("results/runtime_baseline_comparison.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in cols})
    # stash the environment as a comment-friendly sidecar for the report
    np.savez_compressed("results/runtime_baseline_env.npz", env=np.array(E, dtype=object), rows=np.array(rows, dtype=object))
    print(f"\n({clk()-t_start:.0f}s) saved results/runtime_baseline_comparison.csv + runtime_baseline_env.npz")
