"""
FOURTH-ORDER FAIRNESS TEST (PROTOCOL "FOURTH-ORDER CUMULANT BASELINE"). Head-to-head on the superradiant
burst peak: exact vs plain 2nd-order vs kappa4=0 (T3) vs kappa5=0 (T4) vs frozen V7. Same cells/seeds as
Result 24 (POCKET/CONTROL, seeds 50-52) so plain/V7 numbers are directly comparable. Reports SIGNED F_peak
error, STABILITY (blow-up), and COST (build + rollout time). Maps to pre-registered outcomes O1/O2/O3.
"""
import sys
import time
import numpy as np
from scipy.integrate import solve_ivp

from pilot_eval import build_cell, m0_product, run_from, WC
from v7 import run_v7
from local_run import observables, stable_ok
from exact_tc import draw_disorder
from t3_truncation import T3Engine
from t4_truncation import T4Engine
from burst_pred import exact_ref_big, burst_obs, SG, NC_REF

GDOWN = 0.02
NC_TRUNC = 10
REGIMES = {"POCKET": (0.70, 0.03, 0.02), "CONTROL": (0.70, 0.15, 0.02)}
SEEDS = [50, 51, 52]
TMAX, NT = 40.0, 200
SMOKE = "--smoke" in sys.argv


def roll_trunc(eng):
    """Rollout a truncation engine (T3/T4) from the inverted state; return (ts, n(t)) or (ts, None) if unstable."""
    m0 = np.zeros(len(eng.track), complex)
    for i, (p, q, es) in enumerate(eng.track):
        if p == 0 and q == 0 and all(e in (0, 3) for e in es):
            m0[i] = 1.0
    ts = np.linspace(0, TMAX, NT)

    def rhs(t, y):
        n = len(y) // 2
        d = eng.deriv(y[:n] + 1j * y[n:])
        return np.concatenate([d.real, d.imag])

    def blow(t, y):
        return 1.0e4 - float(np.max(np.abs(y)))
    blow.terminal = True; blow.direction = -1
    sol = solve_ivp(rhs, (0, TMAX), np.concatenate([m0.real, m0.imag]), t_eval=ts,
                    rtol=1e-5, atol=1e-7, method="LSODA", events=blow, max_step=0.5)
    if not sol.success or sol.t[-1] < TMAX - 1e-6 or not np.all(np.isfinite(sol.y)):
        return ts, None
    ni = eng.track.index((1, 1, tuple(0 for _ in range(eng.M))))
    return ts, (sol.y[ni] + 1j * sol.y[ni + len(eng.track)]).real


def signed(fp, fex):
    return (fp - fex) / max(abs(fex), 1e-12)


def run_cell(M, gc, ka, gp, seed, v4, v6):
    w = draw_disorder(M, WC, SG, seed, "gauss")
    g = np.full(M, gc / np.sqrt(M))
    eng, _, _ = build_cell(M, gc, ka, gp, SG, seed)
    det = w - WC
    _, nex, _, frac = exact_ref_big(M, w, g, ka, gp, NC_REF)
    ts = np.linspace(0, TMAX, NT)
    fex = burst_obs(ts, nex, ka)["Fpeak"]
    out = {"fex": fex, "frac": frac}

    m0 = m0_product(eng, 1.0)
    _, tp = run_from(eng, m0, None)
    out["plain"] = (signed(burst_obs(ts, observables(eng, tp)[0], ka)["Fpeak"], fex), True) if stable_ok(tp) else (np.nan, False)
    _, t7 = run_v7(eng, m0, det, (gc, ka, gp))
    out["v7"] = (signed(burst_obs(ts, observables(eng, t7)[0], ka)["Fpeak"], fex), True) if stable_ok(t7) else (np.nan, False)

    for tag, Cls in (("t3", T3Engine), ("t4", T4Engine)):
        t0 = time.time()
        e = Cls(M, NC_TRUNC, WC, w, g, kappa=ka, gdown=GDOWN, gphi=gp, Fsafe=6)
        build_s = time.time() - t0
        t0 = time.time()
        _, n = roll_trunc(e)
        roll_s = time.time() - t0
        if n is None:
            out[tag] = (np.nan, False, build_s, roll_s, len(e.track))
        else:
            out[tag] = (signed(burst_obs(ts, n, ka)["Fpeak"], fex), True, build_s, roll_s, len(e.track))
    return out


if __name__ == "__main__":
    import pickle
    v4 = pickle.load(open("results/lc_v3dens.pkl", "rb"))
    v6 = pickle.load(open("results/lc_v6.pkl", "rb"))
    Ms = [3]                                 # M=4 T4 not built: M=3 already settles it; O(M^5) cost only grows
    seeds3 = SEEDS[:1] if SMOKE else SEEDS
    print("FOURTH-ORDER FAIRNESS TEST: exact vs plain(2nd) vs T3(k4=0) vs T4(k5=0) vs V7 -- signed F_peak error\n")
    res = {}
    for M in Ms:
        cells_seeds = seeds3 if M == 3 else SEEDS[:1]        # M=4: one seed (cost)
        print(f"=== M={M} (seeds {cells_seeds}) ===")
        print(f"  {'regime':8s} {'seed':>4s} | {'plain':>7s} {'T3(k4=0)':>9s} {'T4(k5=0)':>9s} {'V7':>7s} | "
              f"{'T4 build/roll(s)':>16s} {'T4 trk':>7s}")
        for name, (gc, ka, gp) in REGIMES.items():
            for sd in cells_seeds:
                r = run_cell(M, gc, ka, gp, sd, v4, v6)
                res[(M, name, sd)] = r
                def cell(x):
                    v = r[x]
                    return f"{v[0]:+.3f}" if (isinstance(v, tuple) and np.isfinite(v[0])) else "UNSTABLE"
                t4b = r["t4"][2] if len(r["t4"]) > 2 else float("nan")
                t4r = r["t4"][3] if len(r["t4"]) > 3 else float("nan")
                t4trk = r["t4"][4] if len(r["t4"]) > 4 else 0
                print(f"  {name:8s} {sd:>4d} | {cell('plain'):>7s} {cell('t3'):>9s} {cell('t4'):>9s} "
                      f"{cell('v7'):>7s} | {t4b:>7.0f}/{t4r:<7.1f} {t4trk:>7d}", flush=True)
        print()
    np.savez_compressed("results/t4_eval.npz", res=np.array(res, dtype=object))
    print("saved results/t4_eval.npz")
