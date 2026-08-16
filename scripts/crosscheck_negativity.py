"""
LOOSE-END CHECK (not part of the reproduction; diagnostic only).

While patching the Julia reproduction script, the QuantumCumulants second-order closure returned NEGATIVE
<a'a>(t) at a strong-coupling probe point (g = 0.5) well outside the evaluated cells. A negative photon number
is unphysical, so it is either

  (A) a genuine positivity violation of the second-order cumulant closure -- a known failure mode of moment
      truncations, harmless here because it is outside the tested cells; or
  (B) a bug in the Julia setup, which would put the reproduction itself in doubt.

This distinguishes them: it runs the PROJECT'S OWN Python second-order engine (src/engines/scalable_engine.py,
the same build+compile path build_cell() uses) at the IDENTICAL parameters, on a SHARED time grid at MATCHED
solver tolerances, and compares n(t) against the Julia output. Two independent implementations agreeing on an
unphysical excursion is (A). Disagreement is (B).

OUTCOME (2026-08-16): (A). Max |python - julia| = 8.11e-10 over all 201 shared grid points, which is solver
tolerance, while the exact mesolve reference stays non-negative throughout. The negative photon number is a
positivity violation of the second-order CLOSURE at strong coupling -- a known failure mode of moment
truncations -- and not a defect in the Julia reproduction script.

Worth noting what this buys beyond closing the loose end: agreement to 8e-10 on a full trajectory in a regime
where the closure is badly broken is a STRONGER test of implementation equivalence than agreement on the burst
cells, because a mismatch in the Hamiltonian, the dissipators, or the initial condition would show up loudly
here. The two codebases encode the same equations.

Run:  julia --project=. scripts/crosscheck_negativity.jl   # writes the Julia CSV
      python run.py scripts/crosscheck_negativity.py        # compares
"""
import os

import numpy as np
import qutip as qt
from scipy.integrate import solve_ivp

from scalable_engine import ScalableEngine
from exact_tc import build_ops, build_H, collapse_ops
from pilot_eval import m0_product
from local_run import observables

# Exactly the Julia probe point (probe 3): strong coupling, small detuning, no dephasing.
M      = 2
WC     = 1.0
W      = np.array([1.0, 1.02])
G      = np.array([0.5, 0.5])
KAPPA  = 0.1
GDOWN  = 0.01
GPHI   = 0.0
NC     = 12
TMAX   = 20.0

# SHARED grid and matched tolerances -- both sides evaluate at exactly these times. (An earlier version of this
# check compared Python's linspace(0,40,200) samples against Julia's saveat=[0,5,10,15,20] and used the two
# codes' different default tolerances; the apparent disagreement that produced was a time-offset artifact.)
TGRID = np.arange(0.0, 20.0 + 1e-9, 0.1)
RTOL, ATOL = 1e-10, 1e-12
JULIA_CSV = os.path.join(os.path.dirname(__file__), "..", "results", "crosscheck_negativity_julia.csv")


def run_python_cumulant(eng, m0):
    """Plain 2nd-order closure on TGRID at the matched tolerances (same rhs as pilot_eval.run_from)."""
    def rhs(t, y):
        n = len(y) // 2
        mm = y[:n] + 1j * y[n:]
        d = eng.deriv_compiled(mm, None)
        return np.concatenate([d.real, d.imag])

    sol = solve_ivp(rhs, (TGRID[0], TGRID[-1]), np.concatenate([m0.real, m0.imag]),
                    t_eval=TGRID, rtol=RTOL, atol=ATOL, method="RK45")
    n = sol.y.shape[0] // 2
    return np.array([sol.y[:n, i] + 1j * sol.y[n:, i] for i in range(sol.y.shape[1])])


def main():
    if not os.path.exists(JULIA_CSV):
        raise SystemExit("missing results/crosscheck_negativity_julia.csv -- run "
                         "`julia --project=. scripts/crosscheck_negativity.jl` first.")
    jul = np.genfromtxt(JULIA_CSV, delimiter=",", names=True)
    julia_t, julia_n = np.asarray(jul["t"], float), np.asarray(jul["n"], float)
    if julia_t.shape != TGRID.shape or np.max(np.abs(julia_t - TGRID)) > 1e-9:
        raise SystemExit("Julia grid does not match TGRID -- regenerate the CSV.")

    # Same engine + compile path build_cell() uses in the real pipeline (validated == dense to ~1e-14).
    eng = ScalableEngine(M, NC, WC, W, G, kappa=KAPPA, gdown=GDOWN, gphi=GPHI, Fsafe=6).compile()
    m0 = m0_product(eng, 1.0)                       # fully inverted, cavity vacuum
    traj = run_python_cumulant(eng, m0)
    ts = TGRID
    n_cum = np.asarray(observables(eng, traj)[0]).real
    JULIA_T, JULIA_N = julia_t, julia_n

    # exact reference at the same parameters, for context
    ops = build_ops(M, NC)
    H = build_H(ops, WC, W, G)
    cops = collapse_ops(ops, kappa=KAPPA, gdown=GDOWN, gphi=GPHI)
    psi0 = qt.tensor(qt.basis(NC, 0), *[qt.basis(2, 0) for _ in range(M)])
    nop = ops["a"].dag() * ops["a"]
    n_exact = np.array(qt.mesolve(H, psi0, ts, c_ops=cops, e_ops=[nop],
                                  options={"progress_bar": False}).expect[0]).real

    print(f"parameters: M={M} wc={WC} w={list(W)} g={list(G)} kappa={KAPPA} gdown={GDOWN} gphi={GPHI}")
    print(f"python 2nd-order cumulant : min n(t) = {n_cum.min():+.5f}  at t = {ts[int(np.argmin(n_cum))]:.2f}")
    print(f"python exact (mesolve)    : min n(t) = {n_exact.min():+.5f}  (must be >= 0)")
    print()
    print("  t     python_cumulant   julia_cumulant      diff      python_exact")
    for t in (0.0, 5.0, 10.0, 15.0, 20.0):
        i = int(np.argmin(np.abs(ts - t)))
        print(f"{ts[i]:5.1f}   {n_cum[i]:+13.6f}   {JULIA_N[i]:+13.6f}   "
              f"{n_cum[i]-JULIA_N[i]:+9.2e}   {n_exact[i]:+11.5f}")

    # verdict -- compared over the WHOLE shared grid, not a handful of points
    maxdiff = float(np.max(np.abs(n_cum - JULIA_N)))
    print()
    print(f"max |python - julia| over all {len(ts)} shared grid points : {maxdiff:.2e}")
    both_negative = (n_cum.min() < -1e-3) and (JULIA_N.min() < -1e-3)
    exact_ok = n_exact.min() >= -1e-8
    print(f"exact stays non-negative                    : {exact_ok}")
    print(f"both closures go negative                   : {both_negative}")
    print()
    if both_negative and maxdiff < 1e-6 and exact_ok:
        print("VERDICT (A): the two independent second-order implementations agree, including on the")
        print("unphysical excursion, while the exact solve stays non-negative. This is a positivity")
        print("violation of the CLOSURE at strong coupling, not a bug in the Julia script.")
    elif maxdiff >= 1e-6:
        print("VERDICT (B): the implementations DISAGREE. Investigate the Julia setup before trusting")
        print("the reproduction.")
    else:
        print("VERDICT: inconclusive -- read the table above.")


if __name__ == "__main__":
    main()
