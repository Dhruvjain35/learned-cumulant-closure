"""
V5/V6 DEV VERDICT on ledger seeds 19/20 (V6 development; PROTOCOL.md seed ledger). Applied mechanically
against the pre-written DECISION TEMPLATE; partial numbers are never evidence -- this script prints one
completed table and one template verdict.

Rows: deep long-coherence pocket (kappa in {0.03, 0.035} x gphi in {0.02, 0.035, 0.05}), far-M (M=9, 10
at the V0 regime vs seeded mcsolve), and the 4 out-of-pocket sanity cells. Models: V3 (frozen incumbent),
V5 (balanced, no memory), V6 (balanced + memory), V4/V4' as DESCRIPTIVE columns only (excluded from the
decision rule by protocol). Every row prints its provenance: seed ledger status, reference type, and
whether the cell's regime tuple was in ANY training pool.
"""
import pickle
import numpy as np

from closure_v3 import ClosureApplyV3, ratio_transform
from closure_v6 import ClosureApplyV6, run_v6, lambdas
from local_run import observables, stable_ok
from pilot_eval import build_cell, m0_product, run_from, exact_ref, nrmse, WC, NC
from campaign_eval import V0REG
from mc_ref import mc_reference
from scalable_engine import ScalableEngine
from exact_tc import draw_disorder

import sys
DEV_SEEDS = [19, 20]
if "--seeds" in sys.argv:
    DEV_SEEDS = [int(s) for s in sys.argv[sys.argv.index("--seeds") + 1].split(",")]
POCKET = [(ka, gp) for ka in (0.03, 0.035) for gp in (0.02, 0.035, 0.05)]
SANITY = [(0.7, 0.15, 0.02, 0.15), (0.7, 0.15, 0.02, 0.4), (0.7, 0.15, 0.3, 0.15), (1.1, 0.15, 0.02, 0.15)]
GC, SG, M5 = 0.7, 0.15, 5


def load(name):
    with open(f"results/lc_{name}.pkl", "rb") as fh:
        return pickle.load(fh)


def run_model(name, lc, eng, det, regime, m0):
    if name == "v6":
        _, traj, _ = run_v6(eng, m0, ClosureApplyV6(lc, eng, det, regime, xform=ratio_transform),
                            lambdas(regime[1], regime[2]))
    else:
        _, traj = run_from(eng, m0, ClosureApplyV3(lc, eng, det, regime, xform=ratio_transform))
    return traj


if __name__ == "__main__":
    models = {"v3": load("v3rat"), "v5": load("v5"), "v6": load("v6"),
              "v4(desc)": load("v3dens"), "v4p(desc)": load("v4p")}   # frozen incumbents read-only
    print("PROVENANCE: seeds 19/20 = V6 development per ledger (never used before this script);")
    print("            pocket refs = exact mesolve; far-M refs = seeded mcsolve (MC err printed);")
    print("            pocket-pool tuple coverage is kappa {0.03,0.04,0.05} x gphi {0.035,0.05,0.065}:")
    print("            the gphi=0.02 column is EXTRAPOLATION for every model (outside both pools);")
    print("            V0-regime tuple is outside every training pool. v4/v4p are descriptive only.\n")

    res = {}
    print("DEEP POCKET (kappa x gphi, M=5): per-cell NRMSE_n or X, seeds 19/20")
    for (ka, gp) in POCKET:
        for sd in DEV_SEEDS:
            eng, w, g = build_cell(M5, GC, ka, gp, SG, sd)
            det = w - WC
            ne, ze, _ = exact_ref(M5, w, g, ka, gp)
            m0 = m0_product(eng, 1.0)
            for nm, lc in models.items():
                traj = run_model("v6" if nm == "v6" else nm, lc, eng, det, (GC, ka, gp), m0)
                ok = stable_ok(traj)
                res[("P", ka, gp, sd, nm)] = (ok, nrmse(observables(eng, traj)[0], ne) if ok else np.inf)
        line = " | ".join(f"{nm}:" + "/".join(
            (f"{res[('P', ka, gp, s, nm)][1]:.2f}" if res[("P", ka, gp, s, nm)][0] else "X")
            for s in DEV_SEEDS) for nm in models)
        print(f"  ka={ka:.3f} gp={gp:.3f}  {line}", flush=True)

    print("\nFAR-M (V0 regime, mcsolve refs):")
    gc0, ka0, gp0, sg0 = V0REG
    for M in (9, 10):
        for sd in DEV_SEEDS:
            w = draw_disorder(M, WC, sg0, sd, "gauss")
            g = np.full(M, gc0 / np.sqrt(M))
            eng = ScalableEngine(M, NC, WC, w, g, kappa=ka0, gdown=0.02, gphi=gp0, Fsafe=6).compile()
            det = w - WC
            _, nref, zref, mcerr = mc_reference(M, NC, w, g, ntraj=200)
            m0 = m0_product(eng, 1.0)
            parts = []
            for nm, lc in models.items():
                traj = run_model("v6" if nm == "v6" else nm, lc, eng, det, (gc0, ka0, gp0), m0)
                ok = stable_ok(traj)
                v = nrmse(observables(eng, traj)[0], nref) if ok else np.inf
                res[("M", M, sd, nm)] = (ok, v)
                parts.append(f"{nm}={'X' if not ok else f'{v:.3f}'}")
            print(f"  M={M} seed {sd} (MCerr {mcerr:.3f}): " + "  ".join(parts), flush=True)

    sd0 = DEV_SEEDS[0]
    print(f"\nSANITY (seed {sd0}):")
    for (gc, ka, gp, sg) in SANITY:
        eng, w, g = build_cell(M5, gc, ka, gp, sg, sd0)
        det = w - WC
        ne, ze, _ = exact_ref(M5, w, g, ka, gp)
        m0 = m0_product(eng, 1.0)
        parts = []
        for nm, lc in models.items():
            traj = run_model("v6" if nm == "v6" else nm, lc, eng, det, (gc, ka, gp), m0)
            ok = stable_ok(traj)
            v = nrmse(observables(eng, traj)[0], ne) if ok else np.inf
            res[("S", gc, ka, gp, sg, nm)] = (ok, v)   # sg WAS missing -> two cells shared a key (bug,
            # disclosed in Result 17; dev medians were over 3 of 4 cells; not decisive either way)
            parts.append(f"{nm}={'X' if not ok else f'{v:.3f}'}")
        print(f"  ({gc},{ka},{gp},{sg}): " + "  ".join(parts), flush=True)

    # ---- MECHANICAL DECISION TEMPLATE (PROTOCOL.md; v4/v4p excluded) ----
    def stab(nm, kind):
        keys = [k for k in res if k[0] == kind and k[-1] == nm]
        return np.mean([res[k][0] for k in keys])
    def med(nm, kind):
        vals = [res[k][1] for k in res if k[0] == kind and k[-1] == nm and res[k][0]]
        return float(np.median(vals)) if vals else np.inf
    print("\nDECISION TEMPLATE (mechanical; v4 columns descriptive, excluded):")
    for nm in ("v3", "v5", "v6"):
        print(f"  {nm}: pocket stab {100*stab(nm,'P'):.0f}% med {med(nm,'P'):.3f} | "
              f"far-M stab {100*stab(nm,'M'):.0f}% med {med(nm,'M'):.3f} | "
              f"sanity med {med(nm,'S'):.3f}")
    v6_beats_v5_pocket = stab("v6", "P") > stab("v5", "P") or (
        stab("v6", "P") == stab("v5", "P") and med("v6", "P") < med("v5", "P"))
    v6_beats_v5_farm = stab("v6", "M") > stab("v5", "M") or (
        stab("v6", "M") == stab("v5", "M") and med("v6", "M") < med("v5", "M"))
    sanity_ok = med("v6", "S") < 1.5 * max(med("v3", "S"), 1e-9)
    v5_beats_v3 = stab("v5", "P") > stab("v3", "P") or (
        stab("v5", "P") == stab("v3", "P") and med("v5", "P") < med("v3", "P"))
    if v6_beats_v5_pocket and v6_beats_v5_farm and sanity_ok:
        print("  -> TEMPLATE 2: memory is the missing ingredient. Freeze V6; final verdict on 21/22/23.")
    elif v6_beats_v5_pocket and sanity_ok:
        print("  -> TEMPLATE 3: memory helps pocket, insufficient for size transfer.")
    elif v5_beats_v3 and not v6_beats_v5_pocket:
        print("  -> TEMPLATE 1: balancing mattered, memory did not.")
    else:
        print("  -> TEMPLATE 4: V6 fails. Negative result; escalation is a separate scope decision.")
    out = ("results/eval_v56_dev.npz" if DEV_SEEDS == [19, 20]
           else f"results/eval_v56_s{'_'.join(map(str, DEV_SEEDS))}.npz")
    np.savez_compressed(out, res=np.array(res, dtype=object))
    print(f"saved {out}")
