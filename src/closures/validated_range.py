"""
A conservative, regime-blind rollout guard for the frozen closure, enforced in code rather than in prose.
(The status string "validated" below is a legacy label for "inside this guard", not a scientific claim.)

WHY THIS EXISTS
---------------
Three separate attempts were made to repair the M >= 12 divergence documented in
positivity_audit.py. All three are recorded in the history and all three failed:

  1. Output gating on physicality (physicality_gate.py). Reduces the blow-up by up to eight
     orders of magnitude but never restores positivity, because the ODE solver commits a step
     past the boundary before any violation is observable.
  2. Margined gating, engaging on approach instead of after the fact. Monotonically WORSE
     (M=16 seed 7: max|moment| 4.8e4 -> 3.7e5 as the margin grows from 0 to 0.3).
  3. Routing to the memoryless branch, on the theory that V6's memory traces accumulate error.
     Refuted: V4 fails too, and at (M=20, seed 9) it is seven orders of magnitude WORSE than
     V6 (4.5e8 vs 52.7). There is no stable branch to route to.

The instability is therefore a property of the learned closure evaluated far outside its
training window, not of any mechanism that can be gated or dispatched around. What remains is
to stop pretending otherwise, and to make the boundary impossible to cross by accident.

WHAT THIS GUARD IS, AND WHAT IT IS NOT (rewritten 2026-10-07; behaviour unchanged)
------------------------------------------------------------------------------------
VALIDATED_MAX_M and KNOWN_UNSTABLE_MIN_M are a conservative, regime-blind SOFTWARE GUARD on full-length
rollouts, set in August 2026 from Results 30-31. They are not a statement of the scientific validity envelope,
which has two parts that a single integer cannot express (FINDINGS Results 30, 34-38; the paper, S10):

  Full-trajectory physicality, regime-dependent (t <= 40):
    low loss (kappa 0.03, gamma_phi 0.02): physical on every tested draw through M = 8; leaks on 1 of 3 draws
      at M = 10; late-time divergence at M >= 12 (1 of 3 draws at M = 12-16, 3 of 3 at M = 20), with or
      without disorder, on both frozen branches.
    normal loss (kappa 0.15): exactly physical through M = 20 at sigma = 0; not measured with disorder above
      M = 10 over the full horizon.
    far-M trajectory check (g_c 0.8, kappa 0.1): the V4 branch was unstable on 1 of 3 draws at M = 9 and 10
      (Result 19) -- so "M <= 10 is always physical" is NOT true either.

  Burst-peak accuracy, referenced beyond M = 10: exact permutation-invariant solves at sigma = 0 to M = 16
    (error grows ~0.8-0.9 points per emitter) and one disordered quantum-trajectory draw at M = 12
    (+5.7% low loss, +4.0% normal loss). In the low-loss sigma = 0 runs the burst peak (t ~ 4) precedes the
    first violation (t ~ 7.3), so burst values outlive the late-time failure; that is exactly why burst-peak
    evaluation at those sizes is a deliberate, separately referenced exception to this guard.

The guard stays conservative on purpose: by default check() refuses every M > 10 (M = 11 as untested,
M >= 12 as known-unstable), because the failure it guards against is silent.

Calling `check(M)` before a rollout turns a silent unphysical trajectory into an explicit,
citable refusal. `mode="warn"` is available for deliberate exploration, but the default is to
raise, because the failure this guards against does not announce itself -- per-site inversions
diverge with opposite signs, so the MEAN inversion still reads near -1 while individual sites sit
at +/-10^4. That is precisely how the defect survived earlier stability checks.
"""

VALIDATED_MAX_M = 10
KNOWN_UNSTABLE_MIN_M = 12


class OutsideValidatedRange(RuntimeError):
    """Raised when a full rollout is requested outside the conservative guard range."""


def status(M):
    """'validated' | 'untested' | 'known-unstable' for a given emitter count."""
    if M <= VALIDATED_MAX_M:
        return "validated"
    if M < KNOWN_UNSTABLE_MIN_M:
        return "untested"
    return "known-unstable"


def check(M, mode="raise"):
    """Gate a rollout on the validated range. Returns the status string.

    mode="raise" (default) refuses anything past the validated range.
    mode="warn"  emits a warning and proceeds, for deliberate exploration.
    mode="silent" returns the status without acting, for callers that report it themselves.
    """
    s = status(M)
    if s == "validated" or mode == "silent":
        return s

    if s == "known-unstable":
        msg = (f"M={M} is past this conservative rollout guard. In the low-loss regime the frozen "
               f"closure is measured to leave the physical set at late times for M>=12 (systematic "
               f"at M=20); the normal-loss regime was physical to M=20 at sigma=0. Full rollouts "
               f"here are not trustworthy by default. See docs/FINDINGS.md Results 30, 34 and "
               f"src/analysis/positivity_audit.py.")
    else:
        msg = (f"M={M} is past the validated range (M<={VALIDATED_MAX_M}) and below the "
               f"measured failure onset (M>={KNOWN_UNSTABLE_MIN_M}). It is untested in either "
               f"direction.")

    if mode == "warn":
        import warnings
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        return s
    raise OutsideValidatedRange(msg)
