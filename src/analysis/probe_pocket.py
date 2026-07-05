"""
Failure-pocket probe (next experiment per review): dense adversarial grid around the observed instability
region kappa in [0.03, 0.05] x gphi around 0.05 (g=0.7, sigma=0.15, M=5), with per-rollout diagnostics so
the follow-up fix comparison (more local training density vs a bounded/gated output mechanism) is chosen
from evidence, not guesswork.

Per cell we record: stability; NRMSE_n vs exact when stable; divergence onset time (first t where the
tracked-moment magnitude exceeds 10, or where integration stopped short); running max |kappa3| emitted by
the closure; fraction of net outputs hitting the training-range clip; worst per-site inversion bound
violation min_t,i Re<sz_i> below -1; solver rhs-evaluation count (stiffness proxy). Seeds 7 AND 8 per cell
(instability may be realization-dependent).
"""
import pickle
import time
import numpy as np

from closure_v3 import ClosureApplyV3, ratio_transform
from local_run import observables, stable_ok
from pilot_eval import build_cell, m0_product, exact_ref, nrmse, WC, NC, TMAX, NT
from scipy.integrate import solve_ivp

GC, SG, M = 0.7, 0.15, 5
KAS = [0.03, 0.035, 0.04, 0.045, 0.05]
GPS = [0.02, 0.035, 0.05, 0.065, 0.08, 0.1]
SEEDS = [7, 8]


def run_instrumented(eng, m0, clo):
    ts = np.linspace(0, TMAX, NT)

    def rhs(t, y):
        n = len(y) // 2
        mm = y[:n] + 1j * y[n:]
        d = eng.deriv_compiled(mm, clo(mm))
        return np.concatenate([d.real, d.imag])

    sol = solve_ivp(rhs, (ts[0], ts[-1]), np.concatenate([m0.real, m0.imag]),
                    t_eval=ts, rtol=1e-4, atol=1e-6, method="RK45")
    n = sol.y.shape[0] // 2
    traj = np.array([sol.y[:n, i] + 1j * sol.y[n:, i] for i in range(sol.y.shape[1])])
    return ts, traj, sol.nfev, sol.status


def diagnostics(eng, ts, traj):
    """onset time of divergence (first |m|>10 or early stop), worst per-site sz undershoot below -1."""
    mags = np.max(np.abs(traj), axis=1) if len(traj) else np.array([np.inf])
    over = np.nonzero(mags > 10)[0]
    if len(over):
        onset = float(ts[over[0]])
    elif len(traj) < NT:
        onset = float(ts[len(traj) - 1])
    else:
        onset = None
    Mm = eng.M
    zc = [eng.tpos[(0, 0, tuple(3 if s == i else 0 for s in range(Mm)))] for i in range(Mm)]
    L = min(len(traj), NT)
    zmin = float(min(traj[t][c].real for t in range(L) for c in zc)) if L else float("nan")
    return onset, zmin


if __name__ == "__main__":
    with open("results/lc_v3rat.pkl", "rb") as fh:
        lc3 = pickle.load(fh)
    print(f"pocket probe: kappa x gphi dense grid at (g={GC}, sigma={SG}), M={M}, seeds {SEEDS}")
    print(f"{'ka':>6s} {'gp':>6s} {'sd':>3s} {'stable':>7s} {'NRMSE_n':>8s} {'onset_t':>8s} "
          f"{'max|k3|':>8s} {'clip%':>6s} {'z_min':>7s} {'nfev':>6s}")
    rows = {}
    for ka in KAS:
        for gp in GPS:
            for sd in SEEDS:
                t0 = time.time()
                eng, w, g = build_cell(M, GC, ka, gp, SG, sd)
                det = w - WC
                clo = ClosureApplyV3(lc3, eng, det, (GC, ka, gp), xform=ratio_transform)
                clo.stats = {}
                ts, traj, nfev, status = run_instrumented(eng, m0_product(eng, 1.0), clo)
                ok = stable_ok(traj)
                if ok:
                    ne, ze, _ = exact_ref(M, w, g, ka, gp)
                    nn = nrmse(observables(eng, traj)[0], ne)
                else:
                    nn = float("inf")
                onset, zmin = diagnostics(eng, ts, traj)
                maxk = max(clo.stats.get("maxk", [0.0]))
                clipf = 100.0 * clo.stats.get("clip_n", 0) / max(clo.stats.get("tot_n", 1), 1)
                rows[(ka, gp, sd)] = dict(ok=ok, nn=nn, onset=onset, maxk=maxk, clipf=clipf,
                                          zmin=zmin, nfev=nfev)
                print(f"{ka:>6.3f} {gp:>6.3f} {sd:>3d} {str(ok):>7s} "
                      f"{nn:>8.3f} {str(onset):>8s} {maxk:>8.3f} {clipf:>6.1f} {zmin:>7.3f} {nfev:>6d}"
                      f"  ({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed("results/probe_pocket.npz", rows=np.array(rows, dtype=object))
    print("saved results/probe_pocket.npz")
