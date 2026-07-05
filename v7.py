"""
V7: the physics-routed closure (PROTOCOL.md V7 PRE-REGISTRATION). Not a new model -- a fixed dispatch
between two PERMANENTLY FROZEN models, decided at t=0 from global physical inputs only:

    V7(kappa, gphi) = V6 (balanced + physics-timed memory)  if kappa <= 0.045 and gphi <= 0.05
                      V4 (density-augmented memoryless)      otherwise

No blending, no mid-trajectory switching, no state/seed/error-based routing. Where V7 routes to V4 its
trajectory is bit-identical to frozen V4's (same code path, same artifact); likewise for V6 -- enforced
by branch_equivalence_check() before any V7 evaluation is trusted.
"""
import pickle
import numpy as np

from closure_v3 import ClosureApplyV3, ratio_transform
from closure_v6 import ClosureApplyV6, run_v6, lambdas
from pilot_eval import run_from

BOX_KA, BOX_GP = 0.045, 0.05
_cache = {}


def _load(name):
    if name not in _cache:
        with open(f"results/lc_{name}.pkl", "rb") as fh:
            _cache[name] = pickle.load(fh)
    return _cache[name]


def route(ka, gp):
    return "v6" if (ka <= BOX_KA and gp <= BOX_GP) else "v4"


def run_v7(eng, m0, det, regime):
    """regime = (gcoll, kappa, gphi). Returns (branch, traj)."""
    gc, ka, gp = regime
    branch = route(ka, gp)
    if branch == "v6":
        lc = _load("v6")
        _, traj, _ = run_v6(eng, m0, ClosureApplyV6(lc, eng, det, regime, xform=ratio_transform),
                            lambdas(ka, gp))
    else:
        lc = _load("v3dens")                       # V4 == frozen lc_v3dens.pkl
        _, traj = run_from(eng, m0, ClosureApplyV3(lc, eng, det, regime, xform=ratio_transform))
    return branch, traj


def branch_equivalence_check(eng_in, m0_in, det_in, regime_in, eng_out, m0_out, det_out, regime_out):
    """One in-box case and one out-of-box case: V7's trajectory must be IDENTICAL to the frozen model's
    own direct rollout. Proves the router adds zero numerics."""
    b_in, t_in = run_v7(eng_in, m0_in, det_in, regime_in)
    assert b_in == "v6", f"in-box case routed to {b_in}"
    lc6 = _load("v6")
    _, t6, _ = run_v6(eng_in, m0_in, ClosureApplyV6(lc6, eng_in, det_in, regime_in,
                                                    xform=ratio_transform), lambdas(regime_in[1], regime_in[2]))
    d6 = float(np.max(np.abs(t_in - t6)))
    b_out, t_out = run_v7(eng_out, m0_out, det_out, regime_out)
    assert b_out == "v4", f"out-of-box case routed to {b_out}"
    lc4 = _load("v3dens")
    _, t4 = run_from(eng_out, m0_out, ClosureApplyV3(lc4, eng_out, det_out, regime_out,
                                                     xform=ratio_transform))
    d4 = float(np.max(np.abs(t_out - t4)))
    return d6, d4
