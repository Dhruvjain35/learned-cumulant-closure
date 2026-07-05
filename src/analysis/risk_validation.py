"""
Support-risk indicator validation on DEV seeds 7/8 (pre-registered numerics in PROTOCOL.md, fixed before
any risk trace was computed): trigger = first integrator call with d/d0 >= 1; material lead = t_onset -
t_trigger >= 2.0 model time units; success = lead >= 2.0 in >= 80% of unstable rollouts AND, on the three
stable Q1 center-regime rollouts (in-support), time-fraction above threshold < 5%.

Risk is logged along the BASE closure's own trajectory (fade disabled) -- the question is whether the
indicator predicts the base model's blowups, with lead time. Also collects per-cell mean/max d/d0 for the
boundary figure's support-distance panel.
"""
import pickle
import time
import numpy as np
from scipy.integrate import solve_ivp

from closure_v3 import ratio_transform
from local_run import stable_ok
from support_risk import SupportModel, SafeApply
from pilot_eval import build_cell, m0_product, WC, NC, TMAX, NT
from probe_pocket import diagnostics, KAS, GPS

M, GC, SG = 5, 0.7, 0.15
DEV_SEEDS = [7, 8]
CENTER = (0.7, 0.15, 0.02, 0.15)
LEAD_REQ, COVER_REQ, FALSE_ALARM_MAX = 2.0, 0.80, 0.05


def run_logged(eng, m0, clo):
    ts = np.linspace(0, TMAX, NT)

    def rhs(t, y):
        clo.current_t = t
        n = len(y) // 2
        mm = y[:n] + 1j * y[n:]
        d = eng.deriv_compiled(mm, clo(mm))
        return np.concatenate([d.real, d.imag])

    sol = solve_ivp(rhs, (ts[0], ts[-1]), np.concatenate([m0.real, m0.imag]),
                    t_eval=ts, rtol=1e-4, atol=1e-6, method="RK45")
    n = sol.y.shape[0] // 2
    traj = np.array([sol.y[:n, i] + 1j * sol.y[n:, i] for i in range(sol.y.shape[1])])
    return ts, traj


def trigger_time(risk_log):
    for t, r in risk_log:
        if r >= 1.0 and t is not None:
            return float(t)
    return None


if __name__ == "__main__":
    with open("results/lc_v3rat.pkl", "rb") as fh:
        lc3 = pickle.load(fh)
    d3 = np.load("results/train_v3.npz", allow_pickle=True)["data"].item()
    t0 = time.time()
    support = SupportModel(lc3, {t: (ratio_transform(X), Y) for t, (X, Y) in d3.items()})
    print(f"support model fitted ({time.time()-t0:.0f}s); d0 per dominant type: "
          f"{ {t: round(v, 2) for t, v in support.d0.items() if t in lc3.nets} }\n", flush=True)

    print("pocket cells (dev seeds 7/8), base closure with risk logging (fade OFF):")
    leads, unstable_n, cell_risk = [], 0, {}
    for ka in KAS:
        for gp in GPS:
            for sd in DEV_SEEDS:
                eng, w, g = build_cell(M, GC, ka, gp, SG, sd)
                clo = SafeApply(lc3, eng, w - WC, (GC, ka, gp), xform=ratio_transform,
                                support=support, fade=False)
                clo.risk_log = []
                ts, traj = run_logged(eng, m0_product(eng, 1.0), clo)
                ok = stable_ok(traj)
                onset, _ = diagnostics(eng, ts, traj)
                ttrig = trigger_time(clo.risk_log)
                rvals = [r for _, r in clo.risk_log]
                cell_risk.setdefault((ka, gp), []).append((max(rvals), np.mean(rvals)))
                if not ok and onset is not None:
                    unstable_n += 1
                    lead = (onset - ttrig) if ttrig is not None else float("-inf")
                    leads.append(lead)
                    print(f"  ka={ka:.3f} gp={gp:.3f} sd={sd}: UNSTABLE onset={onset:5.1f} "
                          f"trigger={'never' if ttrig is None else round(ttrig, 1)} lead={lead:5.1f}",
                          flush=True)

    ok_leads = sum(1 for L in leads if L >= LEAD_REQ)
    print(f"\nlead >= {LEAD_REQ} in {ok_leads}/{unstable_n} unstable rollouts "
          f"({100*ok_leads/max(unstable_n,1):.0f}%; required >= {100*COVER_REQ:.0f}%)")

    print("\nfalse-alarm control: stable Q1 center-regime rollouts (in-support), seeds 7/8/9:")
    fa = []
    for sd in (7, 8, 9):
        eng, w, g = build_cell(M, *CENTER[:3], CENTER[3], sd)
        clo = SafeApply(lc3, eng, w - WC, CENTER[:3], xform=ratio_transform,
                        support=support, fade=False)
        clo.risk_log = []
        ts, traj = run_logged(eng, m0_product(eng, 1.0), clo)
        frac = np.mean([r >= 1.0 for _, r in clo.risk_log])
        fa.append(frac)
        print(f"  seed {sd}: stable={stable_ok(traj)}  time-fraction d/d0>=1: {100*frac:.1f}%", flush=True)

    passed = (ok_leads / max(unstable_n, 1) >= COVER_REQ) and all(f < FALSE_ALARM_MAX for f in fa)
    print(f"\nPRE-REGISTERED VERDICT: {'PASS' if passed else 'FAIL'} "
          f"(coverage {100*ok_leads/max(unstable_n,1):.0f}%, worst false-alarm {100*max(fa):.1f}%)")
    np.savez_compressed("results/risk_validation.npz",
                        leads=np.array(leads), cell_risk=np.array(cell_risk, dtype=object),
                        false_alarm=np.array(fa))
    print("saved results/risk_validation.npz")
