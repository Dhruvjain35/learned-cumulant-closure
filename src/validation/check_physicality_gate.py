"""
Two tests the physicality gate must pass before it is allowed near a reported number.

TEST 1 -- IT CHANGES NOTHING WHERE V7 IS VALIDATED.
The gate is exactly 1 on the physical set, so on any cell whose trajectory stays physical the
gated closure must reproduce frozen V7 bit-for-bit. If it does not, the gate is silently altering
the results the paper claims, and it must be rejected however well it behaves elsewhere. This is
asserted, not eyeballed: max |traj_gated - traj_v7| must be exactly 0.

TEST 2 -- IT REMOVES THE DIVERGENCE WHERE V7 FAILS.
On the cells positivity_audit flagged (M >= 12, long-coherence corner), the gated closure must
stay finite and inside the physical bounds. Accuracy at those sizes is NOT claimed and is not
tested here -- there is no reference. The claim is only that a divergence becomes a bounded,
physical trajectory that degrades toward the plain second-order baseline.

Run:  python src/validation/check_physicality_gate.py [--smoke]
"""
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("engines", "closures", "evaluation", "analysis", "training"):
    sys.path.insert(0, os.path.join(_HERE, "..", sub))

from pilot_eval import m0_product, run_from, WC, NC, GDOWN  # noqa: E402
from scalable_engine import ScalableEngine  # noqa: E402
from exact_tc import draw_disorder  # noqa: E402
from closure_v3 import ClosureApplyV3, ratio_transform  # noqa: E402
from closure_v6 import ClosureApplyV6, run_v6, lambdas  # noqa: E402
from v7 import route, _load  # noqa: E402
from physicality_gate import PhysicalityGate  # noqa: E402

GC, SIGMA = 0.7, 0.15


def build(M, ka, gp, seed):
    w = draw_disorder(M, WC, SIGMA, seed, "gauss")
    g = np.full(M, GC / np.sqrt(M))
    eng = ScalableEngine(M, NC, WC, w, g, kappa=ka, gdown=GDOWN, gphi=gp, Fsafe=6).compile()
    return eng, w, g


def roll(eng, det, regime, gated, eps=1e-3):
    """One rollout, with or without the gate, through whichever branch V7 routes to."""
    gc, ka, gp = regime
    br = route(ka, gp)
    m0 = m0_product(eng, 1.0)
    if br == "v6":
        inner = ClosureApplyV6(_load("v6"), eng, det, regime, xform=ratio_transform)
        clo = PhysicalityGate(inner, eng, eps=eps) if gated else inner
        _, traj, _ = run_v6(eng, m0, clo, lambdas(ka, gp))
    else:
        inner = ClosureApplyV3(_load("v3dens"), eng, det, regime, xform=ratio_transform)
        clo = PhysicalityGate(inner, eng, eps=eps) if gated else inner
        _, traj = run_from(eng, m0, clo)
    return traj, br


def bounds(eng, traj):
    M = eng.M
    zc = [eng.tpos[(0, 0, tuple(3 if s == i else 0 for s in range(M)))] for i in range(M)]
    nc = eng.tpos[(1, 1, tuple([0] * M))]
    z = np.array([[r[c].real for c in zc] for r in traj])
    n = np.array([r[nc].real for r in traj])
    return dict(finite=bool(np.isfinite(traj).all()),
                maxabs=float(np.max(np.abs(traj))) if np.isfinite(traj).all() else float("inf"),
                z_viol=float(max(0.0, np.max(np.abs(z)) - 1.0)),
                n_min=float(n.min()), npts=len(traj))


def main():
    smoke = "--smoke" in sys.argv
    regime = (GC, 0.03, 0.02)          # long-coherence corner: routes to the V6 branch
    ok = True

    print("TEST 1 -- gated == frozen V7 wherever the trajectory stays physical")
    cells = [(3, 7), (5, 7)] if smoke else [(3, 7), (5, 7), (5, 8), (8, 7), (8, 9)]
    for (M, seed) in cells:
        eng, w, g = build(M, 0.03, 0.02, seed)
        det = w - WC
        t_plain, br = roll(eng, det, regime, gated=False)
        t_gate, _ = roll(eng, det, regime, gated=True)
        same = (t_plain.shape == t_gate.shape) and float(np.max(np.abs(t_plain - t_gate))) == 0.0
        b = bounds(eng, t_plain)
        ok &= same
        print(f"  M={M:<3} seed={seed}  branch={br}  physical={b['z_viol']==0 and b['n_min']>=0}"
              f"   identical={same}  {'PASS' if same else 'FAIL'}")

    print()
    print("TEST 2 -- gated stays finite and physical where frozen V7 diverges")
    bad = [(12, 7)] if smoke else [(12, 7), (16, 7), (20, 7), (20, 8), (20, 9)]
    print(f"  {'M':>3} {'seed':>5} {'':>8} {'finite':>7} {'max|m|':>11} {'z viol':>10} "
          f"{'min n':>11} {'npts':>5}")
    for (M, seed) in bad:
        eng, w, g = build(M, 0.03, 0.02, seed)
        det = w - WC
        for label, gated in (("V7", False), ("gated", True)):
            t, br = roll(eng, det, regime, gated=gated)
            b = bounds(eng, t)
            good = b["finite"] and b["z_viol"] <= 1e-9 and b["n_min"] >= -1e-9
            if gated:
                ok &= good
            print(f"  {M:>3} {seed:>5} {label:>8} {str(b['finite']):>7} {b['maxabs']:>11.3e} "
                  f"{b['z_viol']:>10.2e} {b['n_min']:>11.3e} {b['npts']:>5}"
                  f"{'  <-- unphysical' if not good else ''}")

    print()
    print("RESULT:", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
