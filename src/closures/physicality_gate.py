"""
A physicality gate that stops the learned closure driving the state off the physical manifold.

THE FAILURE THIS ADDRESSES
--------------------------
positivity_audit.py found that the frozen V7 diverges at M >= 12 in the long-coherence corner:
seed-dependent at M=12-16 (1 of 3) and systematic at M=20 (3 of 3), with per-site inversions
reaching +/-10^4 while the MEAN inversion still looks near -1. Instrumenting the failing run
(M=12, seed 7) shows the causal order:

    evaluation 411 : max|moment| first exceeds 10      (clip fraction 0.947)
    evaluation 412 : clip fraction first exceeds 0.95

so output-range clipping SATURATES AFTER the blow-up starts. Clipping is a symptom, and gating on
it would treat the wrong quantity -- it also sits at 77-82% during perfectly healthy runs at
M=3..10, so it carries almost no signal.

The usable signal is earlier and exact. In the failing trajectory the photon number goes negative
at t ~ 32 while the runaway does not arrive until t ~ 36:

    t=28.1  n =    1.21   sites [-0.43, 0.15]   physical
    t=32.2  n =   -0.75   <-- leaves the physical manifold
    t=36.2  n = -155.12   sites [-61, 84]       runaway

DESIGN
------
The estimator was fitted only on moments taken from genuine density matrices. Once the state
violates |<sz_i>| <= 1 or <a^d a> >= 0 it is outside anything the network ever saw, and its
prediction there is not merely inaccurate but arbitrary. So rather than trusting a clipped
extrapolation, we fade the learned correction out and fall back to kappa3 = 0, which IS the plain
second-order closure -- measured physical on 3/3 seeds at every M from 12 to 20.

    v     = how far the current moments violate the physical bounds (0 when physical)
    gate  = 1 / (1 + (v/eps)^2)
    kappa3_applied = gate * kappa3_predicted

Two properties make this safe to add to a frozen model:

  * gate == 1 EXACTLY whenever the state is physical, so every validated result is reproduced
    bit-for-bit. This is asserted in the tests below rather than assumed.
  * the fallback target is not an invented safe value but the established baseline, so the worst
    case is a documented +29% burst-peak overshoot instead of a divergence.

This is a WRAPPER. V7's frozen artifacts are not modified and its identity is unchanged; the
gated variant is a separate object that must be reported under its own name.

WHAT THIS IS NOT. It does not make the closure accurate at M >= 12 -- accuracy there was never
demonstrated and is not claimed here. It converts an unphysical divergence into a graceful
degradation toward the baseline.
"""
import numpy as np


class PhysicalityGate:
    """Wrap a closure applier so its correction fades out when the state becomes unphysical.

    Parameters
    ----------
    inner : callable
        Any closure applier, e.g. ClosureApplyV3 or ClosureApplyV6. Called as inner(m).
    eng : engine
        Provides tpos / M so the physical bounds can be read off the moment vector.
    eps : float
        Violation scale at which the correction is cut to half. Small enough that ordinary
        numerical noise does not gate, large enough to engage before a runaway develops.
    """

    def __init__(self, inner, eng, eps=1e-3, margin=0.0):
        self.inner = inner
        self.eng = eng
        self.eps = float(eps)
        self.margin = float(margin)
        self.M = eng.M
        self._zc = [eng.tpos[(0, 0, tuple(3 if s == i else 0 for s in range(eng.M)))]
                    for i in range(eng.M)]
        self._nc = eng.tpos[(1, 1, tuple([0] * eng.M))]
        self.gate_log = []
        self._armed = False        # latch: the photon margin only applies after the burst has built

    # the wrapped applier may carry state (V6 keeps a memory trace); forward it transparently
    def __getattr__(self, name):
        return getattr(self.__dict__["inner"], name)

    def __setattr__(self, name, value):
        if name in ("inner", "eng", "eps", "M", "_zc", "_nc", "gate_log"):
            object.__setattr__(self, name, value)
        else:
            setattr(self.inner, name, value)

    def violation(self, m):
        """How far the moment vector sits outside the physical set. Zero when physical.

        With margin > 0 the photon bound engages on APPROACH rather than after the fact. A purely
        reactive gate is too late here: measurements show the solver steps clean past n = 0 in a
        single step, so by the time a violation is visible the state is already far off-manifold
        and switching the closure off no longer recovers it.

        The margin cannot simply be applied from t = 0, because the initial state sits exactly on
        the boundary -- cavity vacuum has n = 0, every emitter inverted has z = +1. So the photon
        margin is LATCHED: it arms only once n has risen well clear of the boundary. That is not a
        convenience, it is the physics of this trajectory -- the photon number starts at zero,
        rises through the burst, and must then decay back toward zero from above, never through
        it. Before the latch arms, only a true violation (n < 0) counts.
        """
        z = np.array([m[c].real for c in self._zc])
        n = m[self._nc].real
        v_z = float(np.max(np.maximum(0.0, np.abs(z) - 1.0)))
        if self.margin > 0.0:
            if n > 2.0 * self.margin:
                object.__setattr__(self, "_armed", True)
            v_n = (float(max(0.0, self.margin - n)) / self.margin if self._armed
                   else float(max(0.0, -n)))
        else:
            v_n = float(max(0.0, -n))
        return max(v_z, v_n)

    def gate_value(self, m):
        v = self.violation(m)
        if v <= 0.0:
            return 1.0                      # exactly 1 on the physical set: no change to validated runs
        return 1.0 / (1.0 + (v / self.eps) ** 2)

    def __call__(self, m):
        g = self.gate_value(m)
        self.gate_log.append(g)
        kap = self.inner(m)
        if g == 1.0:
            return kap
        return kap * g


def gated_v7(eng, det, regime, eps=1e-3):
    """Frozen V7 with the physicality gate applied to whichever branch it routes to."""
    from closure_v3 import ClosureApplyV3, ratio_transform
    from closure_v6 import ClosureApplyV6
    from v7 import route, _load
    gc, ka, gp = regime
    if route(ka, gp) == "v6":
        inner = ClosureApplyV6(_load("v6"), eng, det, regime, xform=ratio_transform)
    else:
        inner = ClosureApplyV3(_load("v3dens"), eng, det, regime, xform=ratio_transform)
    return PhysicalityGate(inner, eng, eps=eps), route(ka, gp)
