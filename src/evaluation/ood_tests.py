"""
The two out-of-distribution axes the manuscript currently lists as untested.

An outside review asked whether any out-of-distribution tests were run without retraining. The
honest answer at the time was no on two axes, and both are now stated as untested in the paper:

  * INITIAL CONDITION -- every reported V7 run starts from full inversion with an empty cavity.
  * DISORDER FAMILY   -- every detuning is drawn from the Gaussian family used in training,
                         even though draw_disorder also implements lorentz, uniform and bimodal.

This script runs both against exact references, with the frozen V7 and no retraining of any kind.
It is a genuine out-of-distribution test: the model is not adapted, re-fit, or re-tuned, and the
comparison baseline is the same plain second-order closure used everywhere else in the repo.

The result may be that V7 degrades. That is a legitimate outcome and is reported as such -- the
point of the exercise is to replace an admission of ignorance with a measurement, not to produce
a favourable number.

Metrics per cell, all against the exact reference:
    nrmse_n  : normalised RMS error on the photon-number trajectory
    peak_err : relative error on the burst peak, the observable the paper claims
    stable   : did the rollout stay finite and bounded

Run:  python src/evaluation/ood_tests.py [--smoke]
"""
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("engines", "closures", "evaluation", "analysis", "training"):
    sys.path.insert(0, os.path.join(_HERE, "..", sub))

import qutip as qt  # noqa: E402
from exact_tc import build_ops, build_H, collapse_ops, draw_disorder  # noqa: E402
from scalable_engine import ScalableEngine  # noqa: E402
from pilot_eval import m0_product, run_from, nrmse, WC, NC, GDOWN, TMAX, NT  # noqa: E402
from local_run import observables, stable_ok  # noqa: E402
from closure_v3 import ClosureApplyV3, ratio_transform  # noqa: E402
from closure_v6 import ClosureApplyV6, run_v6, lambdas  # noqa: E402
from v7 import route, _load  # noqa: E402

GC, KAPPA, GPHI, SIGMA = 0.7, 0.03, 0.02, 0.15
SEEDS = [7, 8, 9]                     # held out from training by construction


def build_cell_family(M, gc, ka, gp, sg, seed, family):
    """build_cell, but with the disorder family exposed instead of hard-coded to gauss."""
    w = draw_disorder(M, WC, sg, seed, family)
    g = np.full(M, gc / np.sqrt(M))
    eng = ScalableEngine(M, NC, WC, w, g, kappa=ka, gdown=GDOWN, gphi=gp, Fsafe=6).compile()
    return eng, w, g


def exact_ref_family(M, w, g, ka, gp, f=1.0):
    """Exact reference for an arbitrary initial inversion fraction f."""
    ops = build_ops(M, NC)
    H = build_H(ops, WC, w, g)
    cops = collapse_ops(ops, kappa=ka, gdown=GDOWN, gphi=gp)
    if f >= 1.0:
        rho0 = qt.tensor(qt.basis(NC, 0), *[qt.basis(2, 0) for _ in range(M)])
    else:
        up = f * qt.fock_dm(2, 0) + (1 - f) * qt.fock_dm(2, 1)      # index 0 is excited
        rho0 = qt.tensor(qt.fock_dm(NC, 0), *[up for _ in range(M)])
    n = ops["a"].dag() * ops["a"]
    zmean = sum(ops["sz"]) / M
    ts = np.linspace(0, TMAX, NT)
    e = qt.mesolve(H, rho0, ts, c_ops=cops, e_ops=[n, zmean],
                   options={"progress_bar": False}).expect
    return np.array(e[0]).real, np.array(e[1]).real


def run_v7_traj(eng, m0, det, regime):
    """Frozen V7: fixed t=0 dispatch between the two permanently frozen branches."""
    gc, ka, gp = regime
    br = route(ka, gp)
    if br == "v6":
        _, traj, _ = run_v6(eng, m0, ClosureApplyV6(_load("v6"), eng, det, regime,
                                                    xform=ratio_transform), lambdas(ka, gp))
    else:
        _, traj = run_from(eng, m0, ClosureApplyV3(_load("v3dens"), eng, det, regime,
                                                   xform=ratio_transform))
    return br, traj


def evaluate(M, family, f, seed):
    """One cell: exact reference, plain second order, frozen V7."""
    eng, w, g = build_cell_family(M, GC, KAPPA, GPHI, SIGMA, seed, family)
    det = w - WC
    regime = (GC, KAPPA, GPHI)
    ne, ze = exact_ref_family(M, w, g, KAPPA, GPHI, f=f)
    m0 = m0_product(eng, f)

    _, tp = run_from(eng, m0, None)                       # plain 2nd order
    np_, zp = observables(eng, tp)[:2]
    br, t7 = run_v7_traj(eng, m0, det, regime)
    n7, z7 = observables(eng, t7)[:2]

    def peak_err(x):
        return abs(np.max(x) - np.max(ne)) / max(np.max(ne), 1e-12)

    return dict(
        M=M, family=family, f=f, seed=seed, branch=br,
        nrmse_plain=nrmse(np_, ne), nrmse_v7=nrmse(n7, ne),
        peak_plain=peak_err(np_), peak_v7=peak_err(n7),
        stable_plain=bool(stable_ok(tp)), stable_v7=bool(stable_ok(t7)),
    )


def main():
    smoke = "--smoke" in sys.argv
    M = 3
    seeds = SEEDS[:1] if smoke else SEEDS

    # axis 1: initial condition (Gaussian disorder, as trained); axis 2: disorder family (full inversion)
    conditions = [("gauss", 1.00, "baseline (as trained)")]
    conditions += [("gauss", f, f"partial inversion f={f}") for f in (0.75, 0.50)]
    conditions += [(fam, 1.00, f"{fam} disorder") for fam in ("uniform", "lorentz", "bimodal")]
    if smoke:
        conditions = conditions[:3]

    print("Out-of-distribution tests, frozen V7, no retraining")
    print(f"  M={M}, g_c={GC}, kappa={KAPPA}, gphi={GPHI}, sigma={SIGMA}, seeds={seeds}")
    print()
    print(f"  {'condition':<26} {'branch':>6} {'nrmse plain':>12} {'nrmse V7':>9} "
          f"{'peak plain':>11} {'peak V7':>8} {'stable':>7}")

    rows = []
    for family, f, label in conditions:
        per = [evaluate(M, family, f, s) for s in seeds]
        agg = {k: float(np.median([p[k] for p in per]))
               for k in ("nrmse_plain", "nrmse_v7", "peak_plain", "peak_v7")}
        stable = all(p["stable_v7"] for p in per)
        br = per[0]["branch"]
        rows.append(dict(label=label, family=family, f=f, branch=br, stable=stable, **agg))
        better = agg["peak_v7"] < agg["peak_plain"]
        print(f"  {label:<26} {br:>6} {agg['nrmse_plain']:>12.4f} {agg['nrmse_v7']:>9.4f} "
              f"{100*agg['peak_plain']:>10.1f}% {100*agg['peak_v7']:>7.1f}% "
              f"{str(stable):>7}  {'' if better else '<-- worse than plain'}")

    print()
    base = rows[0]
    print(f"  baseline burst-peak error: plain {100*base['peak_plain']:.1f}%  "
          f"V7 {100*base['peak_v7']:.1f}%")
    degraded = [r for r in rows[1:] if r["peak_v7"] > base["peak_v7"] * 2]
    if degraded:
        print("  DEGRADED (>2x the in-distribution V7 peak error):")
        for r in degraded:
            print(f"    {r['label']}: {100*r['peak_v7']:.1f}%")
    else:
        print("  No condition degraded V7 beyond 2x its in-distribution peak error.")
    unstable = [r for r in rows if not r["stable"]]
    if unstable:
        print("  UNSTABLE:", ", ".join(r["label"] for r in unstable))

    os.makedirs("results", exist_ok=True)
    np.savez_compressed("results/ood_tests.npz", rows=np.array(rows, dtype=object))
    print("  saved results/ood_tests.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
