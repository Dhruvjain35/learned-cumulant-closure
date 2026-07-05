"""
Burst-peak comparison, kappa4=0 THIRD-order truncation (T3) rung (PROTOCOL pre-reg: T3, M<=4). Question:
does conventional third-order truncation FIX the ~29% superradiant-burst-peak overshoot of 2nd-order cumulant,
or does it also miss it (=> the learned closure captures a STRUCTURED correction, not just 'go one order up')?
Same cells/seed as the exact kill test; exact mesolve reference at Nc_ref=16.
"""
import time
import numpy as np
from scipy.integrate import solve_ivp

from t3_truncation import T3Engine
from exact_tc import draw_disorder
from pilot_eval import WC, GDOWN
from burst_pred import exact_ref_big, burst_obs, SG, NC_REF, REGIMES

MS = [2, 3, 4]                          # T3 pre-registered cap
SEED = 50
NC_T3 = 12
TMAX, NT = 40.0, 200


def m0_t3(eng):
    m = np.zeros(len(eng.track), complex)
    for i, (p, q, es) in enumerate(eng.track):
        if p == 0 and q == 0 and all(e in (0, 3) for e in es):
            m[i] = 1.0                  # <sz>=+1 (all excited); product-state value
    return m


def n_index(eng):
    tgt = (1, 1, tuple(0 for _ in range(eng.M)))
    return eng.track.index(tgt)


def run_t3(M, gc, ka, gp, seed):
    w = draw_disorder(M, WC, SG, seed, "gauss")
    g = np.full(M, gc / np.sqrt(M))
    eng = T3Engine(M, NC_T3, WC, w, g, kappa=ka, gdown=GDOWN, gphi=gp, Fsafe=6)
    m0 = m0_t3(eng)
    ts = np.linspace(0, TMAX, NT)

    def rhs(t, y):
        n = len(y) // 2
        mm = y[:n] + 1j * y[n:]
        d = eng.deriv(mm)
        return np.concatenate([d.real, d.imag])

    def blowup(t, y):                    # terminal event: stop crawling near a finite-time divergence
        return 1.0e4 - float(np.max(np.abs(y)))
    blowup.terminal = True
    blowup.direction = -1

    sol = solve_ivp(rhs, (0, TMAX), np.concatenate([m0.real, m0.imag]), t_eval=ts,
                    rtol=1e-5, atol=1e-7, method="LSODA", events=blowup, max_step=0.5)
    # unstable if it terminated early (blew up) or produced non-finite values
    if not sol.success or sol.t[-1] < TMAX - 1e-6 or not np.all(np.isfinite(sol.y)):
        return ts, None
    ni = n_index(eng)
    nt = sol.y[ni] + 1j * sol.y[ni + len(eng.track)]
    return ts, nt.real, w, g


if __name__ == "__main__":
    print(f"T3 (kappa4=0) burst-peak rung, seed={SEED}, M={MS}:\n")
    print(f"  {'regime':8s} {'M':>2s} | {'ref F_peak':>10s} {'T3 F_peak':>10s} {'T3 signed':>10s} {'plain(ref)':>10s}")
    t0 = time.time()
    for name, (gc, ka, gp) in REGIMES.items():
        for M in MS:
            ts, res, *wg = run_t3(M, gc, ka, gp, SEED)
            if res is None:
                print(f"  {name:8s} {M:>2d} | T3 UNSTABLE/failed"); continue
            w, g = wg
            _, ne, _, frac = exact_ref_big(M, w, g, ka, gp, NC_REF)
            rb = burst_obs(ts, ne, ka); tb = burst_obs(ts, res, ka)
            signed = (tb['Fpeak'] - rb['Fpeak']) / max(rb['Fpeak'], 1e-12)
            print(f"  {name:8s} {M:>2d} | {rb['Fpeak']:>10.4f} {tb['Fpeak']:>10.4f} {signed:>+10.3f} "
                  f"{'(plain ~+0.29)':>10s}", flush=True)
    print(f"\n({time.time()-t0:.0f}s) INTERPRETATION: if T3 signed error is also large/positive -> third-order")
    print("truncation does NOT fix the burst peak; the learned closure's recovery is a structured correction.")
