"""
The 2x2 fix experiment (per review). Four variants, identical everywhere except the fix under test:

  base   current v3-RATIO (hard clip at training min/max)            -- baseline
  dens   + pocket-local training density only (train_pocket.npz)     -- is sparse coverage the issue?
  gate   + bounded residual only (tanh at 99th-pct robust bounds)    -- is runaway output the issue?
  both   density + gate                                              -- do they fix different parts?

Judged on UNTOUCHED pocket seeds 9 and 10 (seeds 7/8 informed the diagnosis and are excluded from the
verdict), over the same 5x6 pocket grid, plus a 4-cell out-of-pocket sanity set (a fix that regresses the
main result is disqualified). Per pre-registered criteria the winner must improve ALL of: stability rate,
NRMSE-when-stable, saturation fraction, max|kappa3| -- a gate that stabilizes by flattening the correction
but loses accuracy is NOT a solution.
"""
import os
import pickle
import time
import numpy as np

from closure_v3 import LocalClosureV3, ClosureApplyV3, ratio_transform
from local_run import observables, stable_ok
from pilot_eval import build_cell, m0_product, exact_ref, nrmse, WC, NC
from probe_pocket import run_instrumented, diagnostics

GC, SG, M = 0.7, 0.15, 5
KAS = [0.03, 0.035, 0.04, 0.045, 0.05]
GPS = [0.02, 0.035, 0.05, 0.065, 0.08, 0.1]
JUDGE_SEEDS = [9, 10]
SANITY = [(0.7, 0.15, 0.02, 0.15), (0.7, 0.15, 0.02, 0.4), (0.7, 0.15, 0.3, 0.15), (1.1, 0.15, 0.02, 0.15)]


def rbounds(data):
    return {t: np.maximum(np.percentile(np.abs(Y), 99, axis=0), 1e-6) for t, (X, Y) in data.items()}


def get_models():
    d3 = np.load("results/train_v3.npz", allow_pickle=True)["data"].item()
    with open("results/lc_v3rat.pkl", "rb") as fh:
        lc_base = pickle.load(fh)
    pkl = "results/lc_v3dens.pkl"
    dp = np.load("results/train_pocket.npz", allow_pickle=True)["data"].item()
    merged = {}
    for t in sorted(set(d3) | set(dp)):
        Xs = [d[t][0] for d in (d3, dp) if t in d]
        Ys = [d[t][1] for d in (d3, dp) if t in d]
        merged[t] = (np.vstack(Xs), np.vstack(Ys))
    if os.path.exists(pkl):
        with open(pkl, "rb") as fh:
            lc_dens = pickle.load(fh)
    else:
        lc_dens = LocalClosureV3()
        lc_dens.fit({t: (ratio_transform(X), Y) for t, (X, Y) in merged.items()})
        with open(pkl, "wb") as fh:
            pickle.dump(lc_dens, fh)
    return lc_base, lc_dens, rbounds(d3), rbounds(merged)


def make_arm(name, lc_base, lc_dens, rb3, rbm, eng, det, regime):
    if name == "base":
        return ClosureApplyV3(lc_base, eng, det, regime, xform=ratio_transform)
    if name == "dens":
        return ClosureApplyV3(lc_dens, eng, det, regime, xform=ratio_transform)
    if name == "gate":
        return ClosureApplyV3(lc_base, eng, det, regime, xform=ratio_transform,
                              bound_mode="tanh", rbound=rb3)
    return ClosureApplyV3(lc_dens, eng, det, regime, xform=ratio_transform,
                          bound_mode="tanh", rbound=rbm)


if __name__ == "__main__":
    lc_base, lc_dens, rb3, rbm = get_models()
    print("2x2 arms ready (dens model fitted/loaded). Judgment: pocket grid x seeds 9,10 (untouched).\n")
    ARMS = ["base", "dens", "gate", "both"]
    rows = {}
    ref_cache = {}
    for ka in KAS:
        for gp in GPS:
            for sd in JUDGE_SEEDS:
                eng, w, g = build_cell(M, GC, ka, gp, SG, sd)
                det = w - WC
                rk = (ka, gp, sd)
                if rk not in ref_cache:
                    ref_cache[rk] = exact_ref(M, w, g, ka, gp)
                ne, ze, _ = ref_cache[rk]
                for arm in ARMS:
                    clo = make_arm(arm, lc_base, lc_dens, rb3, rbm, eng, det, (GC, ka, gp))
                    clo.stats = {}
                    clo.stats_B = rb3           # COMMON yardstick for all arms (normalized r = |k3|/B)
                    ts, traj, nfev, _ = run_instrumented(eng, m0_product(eng, 1.0), clo)
                    ok = stable_ok(traj)
                    nn = nrmse(observables(eng, traj)[0], ne) if ok else float("inf")
                    maxk = max(clo.stats.get("maxk", [0.0]))
                    rN = max(clo.stats.get("rN", 1), 1)
                    rows[(ka, gp, sd, arm)] = dict(
                        ok=ok, nn=nn, maxk=maxk,
                        p90=100.0 * clo.stats.get("r90", 0) / rN,
                        p99=100.0 * clo.stats.get("r99", 0) / rN,
                        rmax=clo.stats.get("rmax", 0.0))
            line = " | ".join(
                f"{arm}:" + "/".join(("S" if rows[(ka, gp, s, arm)]["ok"] else "X") for s in JUDGE_SEEDS)
                for arm in ARMS)
            print(f"  ka={ka:.3f} gp={gp:.3f}  {line}", flush=True)

    print("\nARM SUMMARY over pocket grid x seeds 9,10 (pre-registered criteria; r = |k3|/B_common):")
    print(f"{'arm':>6s} {'stab%':>6s} {'medNRMSE(st)':>13s} {'P(r>.90)%':>10s} {'P(r>.99)%':>10s} "
          f"{'max r':>7s} {'max|k3|':>8s}")
    for arm in ARMS:
        rs = [rows[k] for k in rows if k[3] == arm]
        stab = 100.0 * np.mean([r["ok"] for r in rs])
        nns = [r["nn"] for r in rs if r["ok"]]
        print(f"{arm:>6s} {stab:>6.1f} {np.median(nns) if nns else float('nan'):>13.3f} "
              f"{np.mean([r['p90'] for r in rs]):>10.1f} {np.mean([r['p99'] for r in rs]):>10.1f} "
              f"{np.max([r['rmax'] for r in rs]):>7.2f} {np.mean([r['maxk'] for r in rs]):>8.3f}")

    print("\nSANITY (out-of-pocket, seed 9; any regression vs base disqualifies the arm):")
    for (gc, ka, gp, sg) in SANITY:
        eng, w, g = build_cell(M, gc, ka, gp, sg, 9)
        det = w - WC
        ne, ze, _ = exact_ref(M, w, g, ka, gp)
        vals = []
        for arm in ARMS:
            clo = make_arm(arm, lc_base, lc_dens, rb3, rbm, eng, det, (gc, ka, gp))
            _, traj, _, _ = run_instrumented(eng, m0_product(eng, 1.0), clo)
            vals.append(f"{arm}={nrmse(observables(eng, traj)[0], ne):.3f}"
                        if stable_ok(traj) else f"{arm}=UNSTABLE")
        print(f"  ({gc},{ka},{gp},{sg}): " + "  ".join(vals), flush=True)
    np.savez_compressed("results/fix2x2.npz", rows=np.array(rows, dtype=object))
    print("saved results/fix2x2.npz")
