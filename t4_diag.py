"""
Is the T4 (kappa5=0) blow-up GENUINE or a numerics artifact? Build T4 once (M=3 pocket seed 50), integrate
with several solvers/tolerances, report where/if it diverges and whether n(t) tracks the burst first. Compare
to exact n(t) and to stable T3. If it blows up under stiff solvers + tight tol too, the instability is real.
"""
import time
import numpy as np
from scipy.integrate import solve_ivp

from exact_tc import draw_disorder
from t3_truncation import T3Engine
from t4_truncation import T4Engine
from burst_pred import exact_ref_big, NC_REF

WC, GDOWN = 1.0, 0.02
M, GC, KA, GP, SG, SEED = 3, 0.70, 0.03, 0.02, 0.15, 50
TMAX, NT = 40.0, 200


def m0_of(eng):
    m = np.zeros(len(eng.track), complex)
    for i, (p, q, es) in enumerate(eng.track):
        if p == 0 and q == 0 and all(e in (0, 3) for e in es):
            m[i] = 1.0
    return m


def integrate(eng, method, rtol, atol, cap=1e6):
    m0 = m0_of(eng)
    ni = eng.track.index((1, 1, tuple(0 for _ in range(eng.M))))
    ts = np.linspace(0, TMAX, NT)

    def rhs(t, y):
        n = len(y) // 2
        d = eng.deriv(y[:n] + 1j * y[n:])
        return np.concatenate([d.real, d.imag])

    def blow(t, y):
        return cap - float(np.max(np.abs(y)))
    blow.terminal = True; blow.direction = -1
    t0 = time.time()
    try:
        sol = solve_ivp(rhs, (0, TMAX), np.concatenate([m0.real, m0.imag]), t_eval=ts,
                        rtol=rtol, atol=atol, method=method, events=blow)
    except Exception as e:
        return dict(ok=False, err=str(e)[:60], t=time.time() - t0)
    tend = sol.t[-1] if len(sol.t) else 0.0
    n = (sol.y[ni] + 1j * sol.y[ni + len(eng.track)]).real if sol.y.shape[1] else np.array([])
    return dict(ok=(sol.success and tend >= TMAX - 1e-6), tend=float(tend),
                maxy=float(np.max(np.abs(sol.y))) if sol.y.size else np.nan,
                n=n, ts=sol.t, t=time.time() - t0)


if __name__ == "__main__":
    w = draw_disorder(M, WC, SG, SEED, "gauss")
    g = np.full(M, GC / np.sqrt(M))
    _, nex, _, _ = exact_ref_big(M, w, g, KA, GP, NC_REF)
    tsx = np.linspace(0, TMAX, NT)
    print(f"exact n(t) at t=1,2,3,5,10: {[round(float(np.interp(t, tsx, nex)), 3) for t in (1,2,3,5,10)]}")
    print(f"exact n_peak={nex.max():.3f} at t={tsx[nex.argmax()]:.1f}\n")

    print("build T3 (kappa4=0)...")
    e3 = T3Engine(M, 10, WC, w, g, kappa=KA, gdown=GDOWN, gphi=GP, Fsafe=6)
    r = integrate(e3, "LSODA", 1e-6, 1e-8)
    print(f"  T3 LSODA: ok={r['ok']} tend={r.get('tend',0):.1f} "
          f"n@(1,2,3,5,10)={[round(float(np.interp(t, r['ts'], r['n'])),3) for t in (1,2,3,5,10)] if r['n'].size else 'NA'}\n")

    print("build T4 (kappa5=0)... (~90s)")
    t0 = time.time()
    e4 = T4Engine(M, 10, WC, w, g, kappa=KA, gdown=GDOWN, gphi=GP, Fsafe=6)
    print(f"  built in {time.time()-t0:.0f}s (tracked={len(e4.track)})\n")
    for method, rtol, atol in [("LSODA", 1e-5, 1e-7), ("LSODA", 1e-8, 1e-10),
                               ("Radau", 1e-6, 1e-8), ("BDF", 1e-6, 1e-8)]:
        r = integrate(e4, method, rtol, atol)
        if "err" in r:
            print(f"  T4 {method:6s} rtol={rtol:.0e}: EXCEPTION {r['err']}")
            continue
        npk = f"n_peak={np.nanmax(r['n']):.3f}" if r['n'].size else ""
        ntr = [round(float(np.interp(t, r['ts'], r['n'])), 3) for t in (1, 2, 3, 5, 10)] if r['n'].size else 'NA'
        print(f"  T4 {method:6s} rtol={rtol:.0e}: ok={r['ok']} tend={r['tend']:.2f} maxy={r['maxy']:.1e} "
              f"{npk}  n@(1,2,3,5,10)={ntr}  ({r['t']:.0f}s)")
    print("\nREAD: if T4 diverges (tend<40, maxy huge) under Radau/BDF + tight tol too, the kappa5=0 instability")
    print("is GENUINE, not a solver artifact. If it tracks exact n(t) early then blows up, it's a real closure")
    print("pathology (finite-time divergence), the strongest form of outcome O3.")
