"""
Gate the permutation-invariant reference against the project's exact solver.

Nothing produced by piqs_reference.py is used in any claim until this passes. The test is the
strongest one available: at zero disorder the two solvers describe the SAME physical model by two
completely different routes -- a full 2^M Hilbert space with per-emitter operators (exact_tc.py,
QuTiP mesolve) versus a polynomially-sized Dicke basis with collective operators (qutip.piqs). They
must agree on the whole photon-number trajectory, not just the burst peak.

Run:  python run.py src/validation/check_piqs_vs_exact.py
"""
import numpy as np
import qutip as qt

from exact_tc import build_ops, build_H, collapse_ops
from piqs_reference import burst_piqs, calibrate_piqs_rates

WC, GC, GDOWN = 1.0, 0.70, 0.02
CELLS = [("POCKET", 0.03, 0.02), ("CONTROL", 0.15, 0.02)]
TMAX, NT, NC = 40.0, 200, 16
TOL = 1e-7


def exact_zero_disorder(M, kappa, gphi):
    ops = build_ops(M, NC)
    w = np.full(M, WC)
    g = np.full(M, GC / np.sqrt(M))
    H = build_H(ops, WC, w, g)
    c = collapse_ops(ops, kappa=kappa, gdown=GDOWN, gphi=gphi)
    psi0 = qt.tensor(qt.basis(NC, 0), *[qt.basis(2, 0)] * M)
    t = np.linspace(0, TMAX, NT)
    res = qt.mesolve(H, psi0, t, c_ops=c, e_ops=[ops["a"].dag() * ops["a"]],
                     options={"progress_bar": False, "atol": 1e-12, "rtol": 1e-10})
    return t, np.asarray(res.expect[0]).real


def main():
    print("=" * 96)
    print("PIQS GATE: permutation-invariant reference vs the project's exact solver, at sigma = 0")
    print("=" * 96)
    print("\nRate-convention calibration (measured, not assumed):")
    factors = calibrate_piqs_rates(verbose=True)
    print(f"   -> emission factor {factors[0]:.4f}, dephasing factor {factors[1]:.4f}"
          "   (PIQS uses spin-1/2 operators, hence the 2)")

    ok = True
    print(f"\n{'regime':<9}{'M':>3}  {'max|n_piqs - n_exact|':>22}  {'peak exact':>11}  {'peak piqs':>11}"
          f"  {'rel peak':>10}   verdict")
    for regime, kappa, gphi in CELLS:
        for M in (2, 3, 4, 5):
            t_e, n_e = exact_zero_disorder(M, kappa, gphi)
            t_p, n_p, fpk, tpk = burst_piqs(M, NC, WC, GC, kappa, GDOWN, gphi,
                                            tmax=TMAX, nt=NT, rate_factors=factors)
            dmax = float(np.max(np.abs(n_p - n_e)))
            pe, pp = float(np.max(n_e)), float(np.max(n_p))
            rel = abs(pp - pe) / pe
            good = dmax < TOL
            ok = ok and good
            print(f"{regime:<9}{M:>3}  {dmax:>22.3e}  {pe:>11.6f}  {pp:>11.6f}  {rel:>10.2e}   "
                  f"{'PASS' if good else 'FAIL'}")
    print("=" * 96)
    print(f"PIQS GATE: {'PASS -- the permutation-invariant reference is trusted at sigma = 0' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
