"""
The validated operating range of the frozen closure, enforced in code rather than in prose.

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

WHAT IS ACTUALLY VALIDATED
--------------------------
  M <= 10   exact or trajectory references exist; V7 is physical on every seed tested and
            reproduces the burst peak to about +/-0.7% on the exact-validated cells.
  M >= 12   no reference exists AND V7 leaves the physical set: seed-dependent at M=12-16
            (1 of 3 seeds) and systematic at M=20 (3 of 3). Both frozen branches fail.

The gap at M=11 is not a claim, it is simply untested.

Calling `check(M)` before a rollout turns a silent unphysical trajectory into an explicit,
citable refusal. `mode="warn"` is available for deliberate exploration, but the default is to
raise, because the failure this guards against does not announce itself -- per-site inversions
diverge with opposite signs, so the MEAN inversion still reads near -1 while individual sites sit
at +/-10^4. That is precisely how the defect survived earlier stability checks.
"""

VALIDATED_MAX_M = 10
KNOWN_UNSTABLE_MIN_M = 12


class OutsideValidatedRange(RuntimeError):
    """Raised when a rollout is requested outside the range the closure is validated on."""


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
        msg = (f"M={M} is outside the validated range and inside the region where the frozen "
               f"closure is measured to diverge (positivity violated at M>=12; systematic at "
               f"M=20). Results here are not merely unvalidated, they are known to leave the "
               f"physical set. See docs/FINDINGS.md and src/analysis/positivity_audit.py.")
    else:
        msg = (f"M={M} is past the validated range (M<={VALIDATED_MAX_M}) and below the "
               f"measured failure onset (M>={KNOWN_UNSTABLE_MIN_M}). It is untested in either "
               f"direction.")

    if mode == "warn":
        import warnings
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        return s
    raise OutsideValidatedRange(msg)
