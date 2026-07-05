"""
Baseline ladder evaluation (PROTOCOL Phase B). Default: dev seeds 29/31 (code verification). With
--seeds 32,33,34: the FINAL all-method comparison. Four regions + far-M, per-region reporting, no pooled
winner. Models: frozen V4, frozen V6 (context), L21/L27/Q21/Q27 (same balanced rows). T3 (kappa4=0) runs
in its own section at M=3/4 (dense engine, gate-passed) with its scaling cost stated.
"""
import sys
import pickle
import time
import numpy as np
from scipy.integrate import solve_ivp

from closure_v3 import ClosureApplyV3, ratio_transform
from closure_v6 import ClosureApplyV6, run_v6, lambdas
from baseline_shell import LocalClosureShell  # noqa: F401 -- required so baseline pickles resolve
from local_run import observables, stable_ok
from pilot_eval import build_cell, m0_product, run_from, exact_ref, nrmse, WC, NC
from campaign_eval import V0REG
from mc_ref import mc_reference
from scalable_engine import ScalableEngine
from exact_tc import build_ops, build_H, collapse_ops, draw_disorder
import qutip as qt

SEEDS = [29, 31]
if "--seeds" in sys.argv:
    SEEDS = [int(s) for s in sys.argv[sys.argv.index("--seeds") + 1].split(",")]
GC, SG, M5 = 0.7, 0.15, 5
POINTS = [  # same four-region set as the V7 verdict
    ("in:deep", 0.030, 0.020), ("in:edge", 0.045, 0.020), ("in:edge", 0.045, 0.050),
    ("in:edge", 0.030, 0.050), ("out:seam", 0.050, 0.020), ("out:seam", 0.050, 0.050),
    ("out:resid", 0.030, 0.065),
]
SANITY = [(0.7, 0.15, 0.02, 0.15), (0.7, 0.15, 0.02, 0.4), (0.7, 0.15, 0.3, 0.15), (1.1, 0.15, 0.02, 0.15)]
MODELS = ["v4", "v6", "l21", "l27", "q21", "q27"]
TRACED = {"v6", "l27", "q27"}
PKL = {"v4": "v3dens", "v6": "v6", "l21": "l21", "l27": "l27", "q21": "q21", "q27": "q27"}
_c = {}


def load(nm):
    if nm not in _c:
        with open(f"results/lc_{PKL[nm]}.pkl", "rb") as fh:
            _c[nm] = pickle.load(fh)
    return _c[nm]


def run_model(nm, eng, det, regime, m0):
    t0 = time.time()
    if nm in TRACED:
        _, traj, _ = run_v6(eng, m0, ClosureApplyV6(load(nm), eng, det, regime, xform=ratio_transform),
                            lambdas(regime[1], regime[2]))
    else:
        _, traj = run_from(eng, m0, ClosureApplyV3(load(nm), eng, det, regime, xform=ratio_transform))
    return traj, time.time() - t0


def fmt(ok, v):
    return f"{v:.3f}" if ok else "X"


if __name__ == "__main__":
    tag = "DEV" if SEEDS == [29, 31] else "FINAL"
    print(f"BASELINE LADDER {tag} on seeds {SEEDS}; models {MODELS} on identical training rows\n")
    res, walltime = {}, {m: [] for m in MODELS}
    print("REGIONS: pocket/seam/residual (M=5); per seed: model=NRMSE or X")
    for (rtag, ka, gp) in POINTS:
        for sd in SEEDS:
            eng, w, g = build_cell(M5, GC, ka, gp, SG, sd)
            det = w - WC
            ne, ze, _ = exact_ref(M5, w, g, ka, gp)
            m0 = m0_product(eng, 1.0)
            parts = []
            for nm in MODELS:
                traj, dt = run_model(nm, eng, det, (GC, ka, gp), m0)
                walltime[nm].append(dt)
                ok = stable_ok(traj)
                v = nrmse(observables(eng, traj)[0], ne) if ok else np.inf
                res[(rtag, ka, gp, sd, nm)] = (ok, v)
                parts.append(f"{nm}={fmt(ok, v)}")
            print(f"  [{rtag}] ka={ka:.3f} gp={gp:.3f} s{sd}: " + " ".join(parts), flush=True)

    print("\nREGION: normal sanity (first seed):")
    for (gc, ka, gp, sg) in SANITY:
        eng, w, g = build_cell(M5, gc, ka, gp, sg, SEEDS[0])
        det = w - WC
        ne, ze, _ = exact_ref(M5, w, g, ka, gp)
        m0 = m0_product(eng, 1.0)
        parts = []
        for nm in MODELS:
            traj, dt = run_model(nm, eng, det, (gc, ka, gp), m0)
            ok = stable_ok(traj)
            v = nrmse(observables(eng, traj)[0], ne) if ok else np.inf
            res[("sanity", gc, ka, gp, sg, nm)] = (ok, v)
            parts.append(f"{nm}={fmt(ok, v)}")
        print(f"  ({gc},{ka},{gp},{sg}): " + " ".join(parts), flush=True)

    print("\nREGION: far-M (V0 regime), models runnable in the scalable engine:")
    gc0, ka0, gp0, sg0 = V0REG
    for M in (9, 10):
        for sd in SEEDS[:2]:
            w = draw_disorder(M, WC, sg0, sd, "gauss")
            g = np.full(M, gc0 / np.sqrt(M))
            eng = ScalableEngine(M, NC, WC, w, g, kappa=ka0, gdown=0.02, gphi=gp0, Fsafe=6).compile()
            det = w - WC
            _, nref, zref, mcerr = mc_reference(M, NC, w, g, ntraj=200)
            m0 = m0_product(eng, 1.0)
            parts = []
            for nm in MODELS:
                traj, dt = run_model(nm, eng, det, (gc0, ka0, gp0), m0)
                ok = stable_ok(traj)
                v = nrmse(observables(eng, traj)[0], nref) if ok else np.inf
                res[("farM", M, sd, nm)] = (ok, v)
                parts.append(f"{nm}={fmt(ok, v)}")
            print(f"  M={M} s{sd} (MCerr {mcerr:.3f}): " + " ".join(parts), flush=True)

    print("\nPER-REGION SUMMARY (stability% | median conditional NRMSE):")
    for want, lab in (("in:", "in-box"), ("out:", "seam+resid"), (("sanity",), "sanity"), (("farM",), "far-M")):
        for nm in MODELS:
            ks = [k for k in res if k[-1] == nm and (k[0].startswith(want) if isinstance(want, str)
                                                     else k[0] in want)]
            st = 100.0 * np.mean([res[k][0] for k in ks])
            vals = [res[k][1] for k in ks if res[k][0]]
            print(f"  {lab:>10s} {nm:>4s}: {st:5.1f}% | {np.median(vals) if vals else float('nan'):.3f}")
        print()
    print("mean rollout wall time (s): " + "  ".join(f"{m}={np.mean(walltime[m]):.1f}" for m in MODELS))
    out = f"results/baselines_{tag.lower()}_s{'_'.join(map(str, SEEDS))}.npz"
    np.savez_compressed(out, res=np.array(res, dtype=object))
    print(f"saved {out}")
