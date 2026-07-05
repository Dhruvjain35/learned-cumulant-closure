"""
V7 evaluation (PROTOCOL.md V7 pre-registration). Default: SEAM TEST on seeds 24/25 (code/route
verification; the box does not move afterward). With --seeds 26,27,28: the FINAL VERDICT, reported as
three regions, never blended. Branch-equivalence asserted before anything else is trusted.

Columns: V7 (with its branch printed per row) plus frozen V4 and V6 run directly at every point --
descriptive context proving V7 equals its chosen branch and showing the road not taken.
"""
import sys
import numpy as np

from closure_v3 import ClosureApplyV3, ratio_transform
from closure_v6 import ClosureApplyV6, run_v6, lambdas
from local_run import observables, stable_ok
from pilot_eval import build_cell, m0_product, run_from, exact_ref, nrmse, WC, NC
from campaign_eval import V0REG
from mc_ref import mc_reference
from scalable_engine import ScalableEngine
from exact_tc import draw_disorder
from v7 import route, run_v7, branch_equivalence_check, _load

SEEDS = [24, 25]
if "--seeds" in sys.argv:
    SEEDS = [int(s) for s in sys.argv[sys.argv.index("--seeds") + 1].split(",")]
GC, SG, M5 = 0.7, 0.15, 5
POINTS = [  # (tag, kappa, gphi) -- pre-registered seam set
    ("deep corner   [in ]", 0.030, 0.020),
    ("inside edge   [in ]", 0.045, 0.020),
    ("inside edge   [in ]", 0.045, 0.050),
    ("inside edge   [in ]", 0.030, 0.050),
    ("outside seam  [out]", 0.050, 0.020),
    ("outside seam  [out]", 0.050, 0.050),
    ("outside seam  [out]", 0.030, 0.065),
]
SANITY = [(0.7, 0.15, 0.02, 0.15), (0.7, 0.15, 0.02, 0.4), (0.7, 0.15, 0.3, 0.15), (1.1, 0.15, 0.02, 0.15)]


def run_direct(name, eng, det, regime, m0):
    if name == "v6":
        _, traj, _ = run_v6(eng, m0, ClosureApplyV6(_load("v6"), eng, det, regime,
                                                    xform=ratio_transform), lambdas(regime[1], regime[2]))
    else:
        _, traj = run_from(eng, m0, ClosureApplyV3(_load("v3dens"), eng, det, regime,
                                                   xform=ratio_transform))
    return traj


def fmt(ok, v):
    return f"{v:.3f}" if ok else "X"


if __name__ == "__main__":
    print(f"V7 {'SEAM TEST' if SEEDS == [24, 25] else 'FINAL VERDICT'} on seeds {SEEDS} "
          f"(ledger; router box kappa<=0.045 & gphi<=0.05, frozen)\n")
    # branch equivalence first: one in-box, one out-of-box case
    eng_i, w_i, _ = build_cell(M5, GC, 0.03, 0.02, SG, SEEDS[0])
    eng_o, w_o, _ = build_cell(M5, GC, 0.15, 0.02, SG, SEEDS[0])
    d6, d4 = branch_equivalence_check(eng_i, m0_product(eng_i, 1.0), w_i - WC, (GC, 0.03, 0.02),
                                      eng_o, m0_product(eng_o, 1.0), w_o - WC, (GC, 0.15, 0.02))
    print(f"branch equivalence: |V7 - V6|_in = {d6:.2e}, |V7 - V4|_out = {d4:.2e}  "
          f"-> {'PASS' if max(d6, d4) == 0.0 else 'FAIL'}\n")
    assert max(d6, d4) == 0.0, "V7 is not bit-identical to its frozen branches"

    res = {}
    print("REGION: pocket + seam (M=5). per-seed NRMSE_n or X;  V7[branch] | v4 | v6")
    for (tag, ka, gp) in POINTS:
        parts = []
        for sd in SEEDS:
            eng, w, g = build_cell(M5, GC, ka, gp, SG, sd)
            det = w - WC
            ne, ze, _ = exact_ref(M5, w, g, ka, gp)
            m0 = m0_product(eng, 1.0)
            br, t7 = run_v7(eng, m0, det, (GC, ka, gp))
            o7 = stable_ok(t7)
            v7v = nrmse(observables(eng, t7)[0], ne) if o7 else np.inf
            t4 = run_direct("v4", eng, det, (GC, ka, gp), m0)
            t6 = run_direct("v6", eng, det, (GC, ka, gp), m0)
            res[("P", ka, gp, sd)] = (o7, v7v, br, stable_ok(t4),
                                      nrmse(observables(eng, t4)[0], ne) if stable_ok(t4) else np.inf,
                                      stable_ok(t6),
                                      nrmse(observables(eng, t6)[0], ne) if stable_ok(t6) else np.inf)
            r = res[("P", ka, gp, sd)]
            parts.append(f"s{sd}: V7[{br}]={fmt(r[0], r[1])} v4={fmt(r[3], r[4])} v6={fmt(r[5], r[6])}")
        print(f"  {tag} ka={ka:.3f} gp={gp:.3f}  " + " | ".join(parts), flush=True)

    print("\nREGION: far-M (V0 regime -> routes to V4 by box):")
    gc0, ka0, gp0, sg0 = V0REG
    for M in (9, 10):
        for sd in SEEDS:
            w = draw_disorder(M, WC, sg0, sd, "gauss")
            g = np.full(M, gc0 / np.sqrt(M))
            eng = ScalableEngine(M, NC, WC, w, g, kappa=ka0, gdown=0.02, gphi=gp0, Fsafe=6).compile()
            det = w - WC
            _, nref, zref, mcerr = mc_reference(M, NC, w, g, ntraj=200)
            m0 = m0_product(eng, 1.0)
            br, t7 = run_v7(eng, m0, det, (gc0, ka0, gp0))
            o7 = stable_ok(t7)
            v7v = nrmse(observables(eng, t7)[0], nref) if o7 else np.inf
            res[("M", M, sd)] = (o7, v7v, br)
            print(f"  M={M} seed {sd} (MCerr {mcerr:.3f}): V7[{br}]={fmt(o7, v7v)}", flush=True)

    print(f"\nREGION: normal sanity (seed {SEEDS[0]}; all route to V4):")
    for (gc, ka, gp, sg) in SANITY:
        eng, w, g = build_cell(M5, gc, ka, gp, sg, SEEDS[0])
        det = w - WC
        ne, ze, _ = exact_ref(M5, w, g, ka, gp)
        m0 = m0_product(eng, 1.0)
        br, t7 = run_v7(eng, m0, det, (gc, ka, gp))
        o7 = stable_ok(t7)
        v7v = nrmse(observables(eng, t7)[0], ne) if o7 else np.inf
        res[("S", gc, ka, gp, sg)] = (o7, v7v, br)
        print(f"  ({gc},{ka},{gp},{sg}): V7[{br}]={fmt(o7, v7v)}", flush=True)

    inb = [res[k] for k in res if k[0] == "P" and route(k[1], k[2]) == "v6"]
    print(f"\nSUMMARY: in-box V7 stability {100*np.mean([r[0] for r in inb]):.0f}% "
          f"med {np.median([r[1] for r in inb if r[0]]):.3f}; "
          f"far-M stability {100*np.mean([res[k][0] for k in res if k[0]=='M']):.0f}%; "
          f"sanity med {np.median([res[k][1] for k in res if k[0]=='S']):.3f}")
    out = f"results/eval_v7_s{'_'.join(map(str, SEEDS))}.npz"
    np.savez_compressed(out, res=np.array(res, dtype=object))
    print(f"saved {out}")
