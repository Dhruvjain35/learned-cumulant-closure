"""
PHYSICS-PREDICTION EXTENSION (PROTOCOL: superradiant burst vs M). Headline question: does the STANDARD
2nd-order cumulant closure's error on a MEASURABLE collective-emission observable grow systematically with
molecule number M in the long-coherence pocket, and does the frozen V7 recover the reference where an exact
reference still exists?

Observables (emitted light = kappa*<a^dag a>(t)), read out of the SAME n(t) each method already produces:
  F_peak = kappa*max_t n(t)      (superradiant burst height)
  t_peak = argmax_t n(t)         (superradiant delay)
  I_tot  = kappa*trapz(n, t)     (integrated emission; peak-finding-robust)

This module runs the CHEAP-FIRST kill test: EXACT mesolve references at M=2..5 (pocket + control), vs plain
cumulant, V7, and (M<=4) the kappa4=0 T3 truncation. Fresh seeds {50,51,52}. mcsolve extension (M=6..12)
is only run (burst_pred_mc.py) if K1/K2 survive here. Pre-registered kill rules K1/K2/K3 evaluated at the end.
"""
import sys
import time
import numpy as np
import qutip as qt
from scipy.integrate import solve_ivp

from pilot_eval import build_cell, m0_product, run_from, WC, GDOWN
from v7 import run_v7
from local_run import observables, stable_ok
from exact_tc import build_ops, build_H, collapse_ops

SG = 0.15
NC_REF = 16                     # reference Fock cutoff (closure engine keeps its frozen NC=12 via build_cell)
FOCK_GUARD = 0.45
SEEDS = [50, 51, 52]
MS_EXACT = [2, 3, 4, 5]
TMAX, NT = 40.0, 200
REGIMES = {                     # (gc, kappa, gphi); gdown=GDOWN, sigma=SG
    "POCKET":  (0.70, 0.03, 0.02),     # inside V7 memory box -> routes to V6; long-coherence, plain fails
    "CONTROL": (0.70, 0.15, 0.02),     # outside box -> routes to V4; negative control, plain should be fine
}
SMOKE = "--smoke" in sys.argv
if SMOKE:
    SEEDS, MS_EXACT = [50], [2, 3]


def exact_ref_big(M, w, g, ka, gp, Nc):
    """Exact mesolve n(t), z(t) at a chosen Fock cutoff Nc (independent of the closure engine's NC)."""
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
    """Peak emitted flux, burst delay, integrated emission from a single n(t)."""
    n = np.asarray(n).real
    ipk = int(np.argmax(n))
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return dict(Fpeak=kappa * float(n[ipk]), tpeak=float(ts[ipk]),
                Itot=kappa * float(_trapz(n, ts)), npeak=float(n[ipk]))


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-12)


def run_cell(M, gc, ka, gp, seed):
    """Return dict of burst observables for reference / plain / v7 (+ meta), or None if Fock-guard drops it."""
    eng, w, g = build_cell(M, gc, ka, gp, SG, seed)
    det = w - WC
    ts, ne, ze, frac = exact_ref_big(M, w, g, ka, gp, NC_REF)
    if frac >= FOCK_GUARD:
        return None, frac
    m0 = m0_product(eng, 1.0)
    out = {"ref": burst_obs(ts, ne, ka)}
    # plain 2nd-order cumulant
    _, tp = run_from(eng, m0, None)
    out["plain"] = burst_obs(ts, observables(eng, tp)[0], ka) if stable_ok(tp) else None
    # V7 (frozen router)
    branch, t7 = run_v7(eng, m0, det, (gc, ka, gp))
    out["v7"] = burst_obs(ts, observables(eng, t7)[0], ka) if stable_ok(t7) else None
    out["branch"] = branch
    return out, frac


if __name__ == "__main__":
    print(f"BURST-vs-M kill test {'(SMOKE) ' if SMOKE else ''}(EXACT mesolve, Nc_ref={NC_REF}); "
          f"seeds={SEEDS}; M={MS_EXACT}\n")
    t0 = time.time()
    data = {}                    # (regime, M, seed) -> (obs, frac)
    for name, (gc, ka, gp) in REGIMES.items():
        for M in MS_EXACT:
            for sd in SEEDS:
                obs, frac = run_cell(M, gc, ka, gp, sd)
                data[(name, M, sd)] = (obs, frac)

    # ---- per-(regime,M) aggregate: relative F_peak error, absolute t_peak error, for plain vs V7 ----
    def agg(name, M, method, key):
        vals = []
        for sd in SEEDS:
            obs, _ = data[(name, M, sd)]
            if obs is None or obs.get(method) is None:
                continue
            r = obs["ref"]; m = obs[method]
            if key == "tpeak":
                vals.append(abs(m["tpeak"] - r["tpeak"]))
            else:
                vals.append(rel(m[key], r[key]))
        return (float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals else (np.nan, np.nan, 0)

    trend = {}                   # (regime, method, key) -> list of (M, mean) for monotonicity check
    for name in REGIMES:
        print(f"=== {name}  (gc,ka,gp)={REGIMES[name]}  branch={data[(name, MS_EXACT[0], SEEDS[0])][0]['branch'] if data[(name, MS_EXACT[0], SEEDS[0])][0] else '?'} ===")
        print(f"  {'M':>2s} | {'plain relF':>11s} {'plain d_tpk':>12s} | {'V7 relF':>9s} {'V7 d_tpk':>9s} | "
              f"{'ref F_peak':>10s} {'ref t_peak':>10s} {'nc/Nc':>6s}")
        for M in MS_EXACT:
            pf, _, npf = agg(name, M, "plain", "Fpeak"); ptp, _, _ = agg(name, M, "plain", "tpeak")
            vf, _, nvf = agg(name, M, "v7", "Fpeak"); vtp, _, _ = agg(name, M, "v7", "tpeak")
            robs = [data[(name, M, sd)][0]["ref"] for sd in SEEDS if data[(name, M, sd)][0]]
            fracs = [data[(name, M, sd)][1] for sd in SEEDS]
            rF = np.mean([o["Fpeak"] for o in robs]) if robs else np.nan
            rT = np.mean([o["tpeak"] for o in robs]) if robs else np.nan
            trend.setdefault((name, "plain", "Fpeak"), []).append((M, pf))
            trend.setdefault((name, "v7", "Fpeak"), []).append((M, vf))
            trend.setdefault((name, "plain", "tpeak"), []).append((M, ptp))
            trend.setdefault((name, "v7", "tpeak"), []).append((M, vtp))
            print(f"  {M:>2d} | {pf:>11.3f} {ptp:>12.3f} | {vf:>9.3f} {vtp:>9.3f} | "
                  f"{rF:>10.3f} {rT:>10.2f} {max(fracs):>6.2f}")
        print()

    # ---- pre-registered kill-rule evaluation ----
    def growing(seq):
        """monotone-ish growth across M: last >= first and last is the max (allow tiny noise)."""
        ys = [y for _, y in seq if np.isfinite(y)]
        if len(ys) < 3:
            return None
        return ys[-1] >= ys[0] and ys[-1] >= max(ys) - 1e-9 and ys[-1] > ys[0] + 1e-6

    pk_plain_F = trend.get(("POCKET", "plain", "Fpeak"), [])
    pk_plain_T = trend.get(("POCKET", "plain", "tpeak"), [])
    pk_v7_F = trend.get(("POCKET", "v7", "Fpeak"), [])
    print("=== PRE-REGISTERED KILL RULES (exact window M<=5) ===")
    gF = growing(pk_plain_F); gT = growing(pk_plain_T)
    print(f"  K1  plain-cumulant error GROWS with M in pocket? F_peak:{gF}  t_peak:{gT}  "
          f"-> {'K1 not triggered (trend present)' if (gF or gT) else 'K1 TRIGGERED (flat/non-monotone) -> null'}")
    # K2: V7 better than plain in pocket (mean over M of pocket F_peak rel err)
    mp = np.nanmean([y for _, y in pk_plain_F]); mv = np.nanmean([y for _, y in pk_v7_F])
    print(f"  K2  V7 F_peak err ({mv:.3f}) < plain ({mp:.3f}) in pocket? "
          f"-> {'K2 not triggered (V7 helps)' if mv < mp else 'K2 TRIGGERED (no V7 burst advantage) -> null'}")
    # K3: control -- plain should be fine and V7 should not hurt
    cp = np.nanmean([y for _, y in trend.get(("CONTROL", "plain", "Fpeak"), [])])
    cv = np.nanmean([y for _, y in trend.get(("CONTROL", "v7", "Fpeak"), [])])
    print(f"  K3  control plain F_peak err={cp:.3f} (want small), V7={cv:.3f} (want <= plain-ish) "
          f"-> {'specificity OK' if (cp < 0.1 and cv <= cp + 0.05) else 'CHECK: specificity weak'}")
    print(f"\nDECISION: extend to mcsolve M=6..12 ONLY if K1 and K2 both survive.")
    np.savez_compressed("results/burst_pred_exact.npz", data=np.array(data, dtype=object),
                        trend=np.array(trend, dtype=object))
    print(f"({time.time()-t0:.0f}s) saved results/burst_pred_exact.npz")
