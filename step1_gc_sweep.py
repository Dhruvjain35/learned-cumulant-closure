"""
STEP 1 (the decider): does the memory-needed boundary move with collective coupling g_c the way the
derived number D_a = 4 g_c / (kappa + gdown + 2 gphi) predicts, with the threshold FROZEN from g_c=0.7?
A (kappa, gphi) rectangle cannot shift with g_c. The criterion predicts s_crit(g_c) = 4 g_c/D_crit - gdown.

Three referee fixes applied:
  1. Grid expanded to BRACKET the predicted boundaries s_crit(0.4)~0.043, (0.7)~0.09, (1.0)~0.137:
     kappa in {0.02..0.08}, gphi in {0.005..0.065} -> s = kappa+2gphi spans ~0.03..0.21.
  2. g_c in {0.4,0.7,1.0} is IN the frozen closures' training grid (GCOLLS=[0.4,0.7,1.0]); g_c is a feature.
     No retraining on Step 1 data. Loss axes go below training mins -> those cells can be
     combined-generalization (class B, both fail), reported separately, NOT counted as memory physics.
  3. Full metric panel (not just AUC): balanced accuracy, false-negative rate (criterion says V4 safe but
     V4 fails -- the dangerous error), false-positive rate (needless routing to V6), observed vs predicted
     50%-propensity contour in s, and boundary-location error |s_crit_pred - s_crit_obs|.

Reference: EXACT mesolve at M=5 with a Fock guard (nc_peak/Nc < 0.45 or the cell is dropped, logged).
Fresh seeds 40,41,42 (disjoint from training rng {5,15,30,...} and every prior eval seed).
"""
import sys
import pickle
import time
import numpy as np
import qutip as qt

from closure_v3 import ClosureApplyV3, ratio_transform
from closure_v6 import ClosureApplyV6, run_v6, lambdas
from local_run import observables, stable_ok
from pilot_eval import build_cell, m0_product, run_from, nrmse, WC
from exact_tc import build_ops, build_H, collapse_ops
from baseline_shell import LocalClosureShell  # noqa: F401
from gen_data_v3 import GCOLLS as TRAIN_GCOLLS

GCS = [0.4, 0.7, 1.0]
KAS = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08]
GPS = [0.005, 0.02, 0.035, 0.05, 0.065]
SEEDS = [40, 41, 42]
GDOWN, M5, SG, NC = 0.02, 5, 0.15, 14        # NC bumped to 14 for the low-loss (bigger burst) cells
ADEQ = 0.15
assert set(GCS) <= set(TRAIN_GCOLLS), f"g_c not in training grid {TRAIN_GCOLLS}"
assert not (set(SEEDS) & {5, 15, 30, 105, 115, 130, 205, 215, 230})
SMOKE = "--smoke" in sys.argv
if SMOKE:
    GCS, KAS, GPS, SEEDS = [0.7], [0.02, 0.03, 0.05], [0.005, 0.02, 0.05], [40]


def load(n):
    return pickle.load(open(f"results/lc_{n}.pkl", "rb"))


def exact_ref_guarded(M, w, g, ka, gp):
    """Exact mesolve obs + Fock-peak guard. Returns (n_ref, z_ref, nc_peak/Nc)."""
    ops = build_ops(M, NC)
    H = build_H(ops, WC, w, g)
    cops = collapse_ops(ops, kappa=ka, gdown=GDOWN, gphi=gp)
    psi0 = qt.tensor(qt.basis(NC, 0), *[qt.basis(2, 0) for _ in range(M)])
    n = ops["a"].dag() * ops["a"]; zmean = sum(ops["sz"]) / M
    ts = np.linspace(0, 40.0, 200)
    e = qt.mesolve(H, psi0, ts, c_ops=cops, e_ops=[n, zmean], options={"progress_bar": False}).expect
    nref = np.array(e[0]).real
    return nref, np.array(e[1]).real, float(nref.max()) / NC


def adequate(ok, nn):
    return bool(ok) and np.isfinite(nn) and nn <= ADEQ


def s_of(ka, gp):
    return ka + 2 * gp


if __name__ == "__main__":
    v4, v6 = load("v3dens"), load("v6")
    print(f"STEP 1 {'(SMOKE)' if SMOKE else ''}: g_c={GCS} (all in training grid {TRAIN_GCOLLS}); "
          f"kappa={KAS}; gphi={GPS}; seeds={SEEDS}; exact mesolve M={M5}, Nc={NC}\n")
    res, dropped = {}, []
    t0 = time.time()
    for gc in GCS:
        for ka in KAS:
            for gp in GPS:
                for sd in SEEDS:
                    eng, w, g = build_cell(M5, gc, ka, gp, SG, sd)
                    det = w - WC
                    nref, zref, frac = exact_ref_guarded(M5, w, g, ka, gp)
                    if frac >= 0.45:
                        dropped.append((gc, ka, gp, sd, round(frac, 2)))
                        continue
                    m0 = m0_product(eng, 1.0)
                    _, t4 = run_from(eng, m0, ClosureApplyV3(v4, eng, det, (gc, ka, gp), xform=ratio_transform))
                    _, t6, _ = run_v6(eng, m0, ClosureApplyV6(v6, eng, det, (gc, ka, gp), xform=ratio_transform), lambdas(ka, gp))
                    v4ok = stable_ok(t4); v6ok = stable_ok(t6)
                    v4nn = nrmse(observables(eng, t4)[0], nref) if v4ok else np.inf
                    v6nn = nrmse(observables(eng, t6)[0], nref) if v6ok else np.inf
                    v4a, v6a = adequate(v4ok, v4nn), adequate(v6ok, v6nn)
                    lab = 'M' if (not v4a and v6a) else ('F' if v4a else 'B')
                    # store the FULL raw record, not just the label
                    res[(gc, ka, gp, sd)] = dict(lab=lab, s=s_of(ka, gp), v4ok=v4ok, v4nn=v4nn,
                                                 v6ok=v6ok, v6nn=v6nn, ncfrac=round(frac, 3))
        print(f"g_c={gc}: cell propensity M/(M+F)  [B=both-fail shown as (bN)]")
        for ka in KAS:
            cells = []
            for gp in GPS:
                labs = [res[(gc, ka, gp, sd)]['lab'] for sd in SEEDS if (gc, ka, gp, sd) in res]
                m = labs.count('M'); f = labs.count('F'); b = labs.count('B')
                tag = f"{m}/{m+f}" + (f"(b{b})" if b else "")
                cells.append(f"s{s_of(ka,gp):.3f}:{tag}")
            print(f"   ka={ka:.2f}  " + "  ".join(cells), flush=True)
    if dropped:
        print(f"\nFock-guard dropped {len(dropped)} cells (nc_peak/Nc>=0.45): {dropped[:6]}{'...' if len(dropped)>6 else ''}")

    # RAW PANEL first (per PROTOCOL wording lock): every (g_c,kappa,gphi,seed) row, written to CSV.
    import csv
    with open("results/step1_raw_panel.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["g_c", "kappa", "gphi", "seed", "s", "predicted_route", "V4_stable", "V4_exact_err",
                     "V6_stable", "V6_exact_err", "observed_label", "nc_peak_frac"])
        for (gc, ka, gp, sd), r in sorted(res.items()):
            wr.writerow([gc, ka, gp, sd, f"{r['s']:.3f}", "", int(r['v4ok']),
                         f"{r['v4nn']:.4f}" if np.isfinite(r['v4nn']) else "UNSTABLE", int(r['v6ok']),
                         f"{r['v6nn']:.4f}" if np.isfinite(r['v6nn']) else "UNSTABLE", r['lab'], r['ncfrac']])
    print("\nraw per-seed panel written to results/step1_raw_panel.csv "
          f"({len(res)} rows). B(both-fail) count: {sum(r['lab']=='B' for r in res.values())}")

    if not SMOKE:
        def contour(gc):
            """observed 50% memory-need contour in s (cell-level, M-vs-F only)."""
            pts = []
            for ka in KAS:
                for gp in GPS:
                    labs = [res[(gc, ka, gp, sd)]['lab'] for sd in SEEDS if (gc, ka, gp, sd) in res and res[(gc, ka, gp, sd)]['lab'] in ('M', 'F')]
                    if labs:
                        pts.append((s_of(ka, gp), labs.count('M') / len(labs)))
            pts.sort()
            hi = [s for s, p in pts if p >= 0.5]      # cells that need memory (>=50% of seeds)
            lo = [s for s, p in pts if p < 0.5]
            if not hi:
                return 0.0                             # never needs memory at this g_c
            if not lo:
                return float('inf')                    # always needs memory in this grid
            s_hi = max(hi)                             # deepest-into-fine cell still needing memory
            above = [s for s in lo if s > s_hi]        # first fine cell beyond it
            return 0.5 * (s_hi + min(above)) if above else s_hi
        s07 = contour(0.7)
        print("\n=== FROZEN-CRITERION COLLAPSE TEST (D_crit calibrated on g_c=0.7 ONLY) ===")
        if s07 is None:
            print("  g_c=0.7 has no clean propensity crossing -> inconclusive")
        else:
            Dc = 4 * 0.7 / (s07 + GDOWN)
            print(f"  calibrated: s_crit(0.7)={s07:.3f} -> D_crit={Dc:.1f}\n")
            print(f"  {'g_c':>4s} {'s_pred':>7s} {'s_obs':>7s} {'|err|':>6s} {'balAcc':>7s} {'FNR':>5s} {'FPR':>5s}")
            for gc in GCS:
                s_pred = 4 * gc / Dc - GDOWN
                s_obs = contour(gc)
                # frozen-criterion classification on this g_c (predict M iff s < s_pred)
                mf = [(s_of(ka, gp), res[(gc, ka, gp, sd)]['lab']) for ka in KAS for gp in GPS for sd in SEEDS
                      if (gc, ka, gp, sd) in res and res[(gc, ka, gp, sd)]['lab'] in ('M', 'F')]
                tp = sum(s < s_pred and l == 'M' for s, l in mf); fn = sum(s >= s_pred and l == 'M' for s, l in mf)
                tn = sum(s >= s_pred and l == 'F' for s, l in mf); fp = sum(s < s_pred and l == 'F' for s, l in mf)
                sens = tp / (tp + fn) if tp + fn else float('nan'); spec = tn / (tn + fp) if tn + fp else float('nan')
                ba = 0.5 * (sens + spec)
                fnr = fn / (tp + fn) if tp + fn else float('nan'); fpr = fp / (tn + fp) if tn + fp else float('nan')
                err = abs(s_obs - s_pred) if s_obs is not None else float('nan')
                print(f"  {gc:>4.1f} {s_pred:>7.3f} {s_obs if s_obs else float('nan'):>7.3f} {err:>6.3f} "
                      f"{ba:>7.2f} {fnr:>5.2f} {fpr:>5.2f}", flush=True)
            print("\n  SUCCESS: g_c=0.4 and 1.0 observed s_crit track the FROZEN prediction (|err| small,")
            print("  direction correct: weaker g_c -> smaller s_crit). KILL: no shift / wrong direction.")
    np.savez_compressed("results/step1_gc_sweep.npz", res=np.array(res, dtype=object),
                        dropped=np.array(dropped, dtype=object))
    print(f"\n({time.time()-t0:.0f}s) saved results/step1_gc_sweep.npz")
