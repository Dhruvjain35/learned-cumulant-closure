"""
GATE 2 for the MPS baseline: does the MPS trajectory match exact mesolve?

Gate 1 (check_mps_liouvillian.py) proved the generator is right. This one proves the *evolution*
is right: same model, same initial state, same observables, compared against QuTiP's exact
master-equation solve on every cell where the exact solve is affordable.

The gate is deliberately strict. The MPS is a controlled approximation with two knobs (bond
dimension chi and time step dt), so at large chi and small dt it must converge to the exact
answer. If it does not, either the generator or the evolution is wrong, and no timing or accuracy
number from this solver means anything.

Reported per cell:
  - max |n_mps(t) - n_exact(t)| over the trajectory
  - relative error on the burst peak, which is the observable the paper actually claims
  - Tr(rho) drift, which should stay at 1

Run:  python src/validation/check_mps_vs_exact.py
      python src/validation/check_mps_vs_exact.py --smoke
"""
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "baselines"))
sys.path.insert(0, os.path.join(_HERE, "..", "engines"))

import qutip as qt  # noqa: E402
from mps_lindblad import run_mps  # noqa: E402
from exact_tc import build_ops, build_H, collapse_ops, draw_disorder  # noqa: E402

WC, GDOWN = 1.0, 0.02
GC, KAPPA, GPHI, SIGMA = 0.7, 0.03, 0.02, 0.15
TMAX, NT = 20.0, 101


def exact_traj(M, Nc, w, g, kappa, gphi, tmax, nt):
    """Exact photon-number and inversion trajectories from QuTiP mesolve."""
    ops = build_ops(M, Nc)
    H = build_H(ops, WC, w, g)
    c = collapse_ops(ops, kappa=kappa, gphi=gphi, gdown=GDOWN)
    psi0 = qt.tensor(qt.basis(Nc, 0), *[qt.basis(2, 0) for _ in range(M)])  # vacuum + all excited
    ts = np.linspace(0.0, tmax, nt)
    n_op = ops["a"].dag() * ops["a"]
    z_ops = ops["sz"]
    t0 = time.time()
    res = qt.mesolve(H, psi0, ts, c_ops=c, e_ops=[n_op] + z_ops,
                     options={"atol": 1e-10, "rtol": 1e-8})
    secs = time.time() - t0
    n = np.array(res.expect[0]).real
    z = np.mean([np.array(res.expect[1 + i]).real for i in range(M)], axis=0)
    return ts, n, z, secs


def peak(n):
    return float(np.max(n))


def main():
    smoke = "--smoke" in sys.argv
    cells = [(2, 6, 48), (3, 8, 96)] if smoke else [(2, 6, 64), (3, 8, 128), (4, 10, 200)]
    tmax, nt = (10.0, 51) if smoke else (TMAX, NT)

    print(f"MPS vs exact mesolve   g_c={GC} kappa={KAPPA} gphi={GPHI} sigma={SIGMA} "
          f"gdown={GDOWN}  t<={tmax}")
    print(f"{'M':>2} {'Nc':>3} {'chi':>5} {'max|dn|':>10} {'peak_exact':>11} {'peak_mps':>10} "
          f"{'rel_peak':>9} {'TrDrift':>9} {'t_exact':>8} {'t_mps':>7}")

    ok = True
    rows = []
    for (M, Nc, chi) in cells:
        w = draw_disorder(M, WC, SIGMA, seed=7, family="gauss")
        g = np.full(M, GC / np.sqrt(M))
        ts, ne, ze, se = exact_traj(M, Nc, w, g, KAPPA, GPHI, tmax, nt)
        r = run_mps(M, Nc, WC, w, g, KAPPA, GDOWN, GPHI, tmax=tmax, nt=nt, chi_max=chi)
        dn = float(np.max(np.abs(r["n"] - ne)))
        pe, pm = peak(ne), peak(r["n"])
        rel = abs(pm - pe) / max(pe, 1e-12)
        drift = float(np.max(np.abs(r["trace"] - 1.0)))
        good = (dn < 5e-3) and (rel < 5e-3) and (drift < 1e-6)
        ok &= good
        rows.append(dict(M=M, Nc=Nc, chi=chi, max_dn=dn, peak_exact=pe, peak_mps=pm,
                         rel_peak=rel, trace_drift=drift, t_exact=se, t_mps=r["seconds"],
                         chi_reached=r["chi_max_reached"]))
        print(f"{M:>2} {Nc:>3} {chi:>5} {dn:>10.2e} {pe:>11.6f} {pm:>10.6f} "
              f"{rel:>9.2e} {drift:>9.1e} {se:>8.1f} {r['seconds']:>7.1f}  "
              f"{'PASS' if good else 'FAIL'}")

    os.makedirs("results", exist_ok=True)
    np.savez("results/mps_gate.npz", rows=np.array(rows, dtype=object))
    print("\nGATE 2", "PASSED" if ok else "FAILED",
          "  (thresholds: max|dn|<5e-3, rel_peak<5e-3, Tr drift<1e-6)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
