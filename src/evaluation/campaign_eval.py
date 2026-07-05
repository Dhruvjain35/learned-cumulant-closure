"""
Campaign evaluation. Answers EXACTLY four pre-registered questions (PROTOCOL.md; pilot locked at tag
pilot-v3 chose the RATIO representation):

  Q1  Does the ratio model win across SEEDS (7,8,9; paired, same realizations for every method),
      not just representative traces? (+ the IC f=0.75 stress across the same seeds)
  Q2  Does it keep winning in the hard regions: low kappa, high sigma, high gphi? Two exact-referenced
      heatmaps at M=5 with cells outside the training hull flagged EXTRAP; paired bootstrap CIs of the
      median NRMSE_n improvement, interpolation and extrapolation cells SEPARATELY.
  Q3  Does size transfer survive at M=8 and M=10 against exact (seeded mcsolve) references, at the V0
      regime so the frozen v2 numbers anchor the comparison?
  Q4  Absolute errors reported everywhere (NRMSE, max-abs, peak error), never ratios alone.

Methods: plain 2nd-order cumulant | frozen v2 (lc_new.pkl) | v3-RATIO (trained once here on the full
648-config grid, pickled). Mean-field: identically frozen for these incoherent inverted ICs (documented
physics; not rerun per cell).
"""
import os
import pickle
import time
import numpy as np

from closure_v3 import LocalClosureV3, ClosureApplyV3, ratio_transform
from local_run import ClosureApply, observables, stable_ok
from pilot_eval import build_cell, m0_product, run_from, exact_ref, nrmse, WC, NC, TMAX, NT
from mc_ref import mc_reference
from exact_tc import draw_disorder

CENTER = (0.70, 0.15, 0.02, 0.15)              # (gcoll, kappa, gphi, sigma)
V0REG = (0.80, 0.10, 0.02, 0.15)               # the frozen-V0 regime (anchors v2 comparison)
SEEDS_Q1 = [7, 8, 9]
# training hull (gen_data_v3): gcoll [0.4,1.0], kappa [0.05,0.3], gphi [0.02,0.1], sigma [0.05,0.3]
HULL = {"gc": (0.4, 1.0), "ka": (0.05, 0.3), "gp": (0.02, 0.1), "sg": (0.05, 0.3)}
MAP_A = ([0.3, 0.4, 0.55, 0.7, 0.85, 1.0, 1.1],          # gcoll  (x kappa=0.15, gphi=0.02)
         [0.0, 0.05, 0.15, 0.25, 0.35, 0.4])              # sigma
MAP_B = ([0.03, 0.05, 0.1, 0.15, 0.3, 0.4],               # kappa  (x gcoll=0.7, sigma=0.15)
         [0.01, 0.02, 0.05, 0.1, 0.2, 0.3])                # gphi


def in_hull(gc, ka, gp, sg):
    return (HULL["gc"][0] <= gc <= HULL["gc"][1] and HULL["ka"][0] <= ka <= HULL["ka"][1]
            and HULL["gp"][0] <= gp <= HULL["gp"][1] and HULL["sg"][0] <= sg <= HULL["sg"][1])


def get_v3rat():
    pkl = "results/lc_v3rat.pkl"
    if os.path.exists(pkl):
        with open(pkl, "rb") as fh:
            return pickle.load(fh)
    d = np.load("results/train_v3.npz", allow_pickle=True)["data"].item()
    lc = LocalClosureV3()
    lc.fit({t: (ratio_transform(X), Y) for t, (X, Y) in d.items()})
    with open(pkl, "wb") as fh:
        pickle.dump(lc, fh)
    return lc


def metrics(eng, traj, ne, ze):
    """(NRMSE_n, NRMSE_z, maxabs, peak_err, stable) -- Q4: absolutes everywhere."""
    if not stable_ok(traj):
        return dict(nn=np.inf, nz=np.inf, mab=np.inf, pk=np.inf, ok=False)
    n, z = observables(eng, traj)
    L = min(len(n), len(ne))
    mab = max(float(np.max(np.abs(n[:L] - ne[:L]))), float(np.max(np.abs(z[:L] - ze[:L]))))
    return dict(nn=nrmse(n, ne), nz=nrmse(z, ze), mab=mab,
                pk=float(abs(n[:L].max() - ne[:L].max())), ok=True)


def eval_cell(lc2, lc3, gc, ka, gp, sg, seed, f=1.0, methods=("plain", "v2", "v3rat")):
    eng, w, g = build_cell(5, gc, ka, gp, sg, seed)
    det = w - WC
    ne, ze, npk = exact_ref(5, w, g, ka, gp, f=f)
    m0 = m0_product(eng, f)
    out = {"nc_frac": npk / NC}
    clos = {"plain": None}
    if "v2" in methods:
        clos["v2"] = ClosureApply(lc2, eng, det)
    if "v3rat" in methods:
        clos["v3rat"] = ClosureApplyV3(lc3, eng, det, (gc, ka, gp), xform=ratio_transform)
    for name, c in clos.items():
        _, traj = run_from(eng, m0, c)
        out[name] = metrics(eng, traj, ne, ze)
    return out


def boot_ci(diffs, nboot=10000, seed=1):
    rng = np.random.default_rng(seed)
    d = np.asarray(diffs, float)
    meds = np.median(rng.choice(d, size=(nboot, len(d)), replace=True), axis=1)
    return float(np.median(d)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def fmt(m):
    return (f"nn={m['nn']:.3f} nz={m['nz']:.3f} mab={m['mab']:.3f} pk={m['pk']:.3f}"
            if m["ok"] else "UNSTABLE")


if __name__ == "__main__":
    with open("results/lc_new.pkl", "rb") as fh:
        lc2 = pickle.load(fh)
    t0 = time.time()
    lc3 = get_v3rat()
    print(f"v3-RATIO trained/loaded ({time.time()-t0:.0f}s); v2 frozen\n", flush=True)

    # ---------- Q1: seeds (paired) ----------
    print("Q1: center regime, M=5, seeds 7/8/9 (paired; same realization per method). Q4: absolutes.")
    q1 = {}
    for sd in SEEDS_Q1:
        r = eval_cell(lc2, lc3, *CENTER, sd)
        q1[sd] = r
        print(f"  seed {sd}: plain[{fmt(r['plain'])}]  v2[{fmt(r['v2'])}]  v3rat[{fmt(r['v3rat'])}]",
              flush=True)
    d = [q1[s]["plain"]["nn"] - q1[s]["v3rat"]["nn"] for s in SEEDS_Q1]
    print(f"  paired NRMSE_n improvement per seed: {[f'{x:.3f}' for x in d]}  (all>0: {all(x>0 for x in d)})")
    print("Q1b: IC f=0.75 stress across the same seeds:")
    q1b = {}
    for sd in SEEDS_Q1:
        r = eval_cell(lc2, lc3, *CENTER, sd, f=0.75)
        q1b[sd] = r
        print(f"  seed {sd}: plain[{fmt(r['plain'])}]  v2[{fmt(r['v2'])}]  v3rat[{fmt(r['v3rat'])}]",
              flush=True)

    # ---------- Q2: heatmaps (plain vs v3rat; per-cell; EXTRAP flagged) ----------
    print("\nQ2 map A: gcoll x sigma at (kappa=0.15, gphi=0.02), M=5, seed 7. cell: plainNN/v3NN flag")
    cells = {}
    for gc in MAP_A[0]:
        rowtxt = []
        for sg in MAP_A[1]:
            r = eval_cell(lc2, lc3, gc, 0.15, 0.02, sg, 7, methods=("plain", "v3rat"))
            tag = "I" if in_hull(gc, 0.15, 0.02, sg) else "E"
            cells[("A", gc, sg)] = (r, tag)
            rowtxt.append(f"{r['plain']['nn']:.2f}/{r['v3rat']['nn']:.2f}{tag}")
        print(f"  gc={gc:4.2f}: " + "  ".join(rowtxt), flush=True)
    print("Q2 map B: kappa x gphi at (gcoll=0.7, sigma=0.15), M=5, seed 7")
    for ka in MAP_B[0]:
        rowtxt = []
        for gp in MAP_B[1]:
            r = eval_cell(lc2, lc3, 0.7, ka, gp, 0.15, 7, methods=("plain", "v3rat"))
            tag = "I" if in_hull(0.7, ka, gp, 0.15) else "E"
            cells[("B", ka, gp)] = (r, tag)
            rowtxt.append(f"{r['plain']['nn']:.2f}/{r['v3rat']['nn']:.2f}{tag}")
        print(f"  ka={ka:4.2f}: " + "  ".join(rowtxt), flush=True)
    for want, lab in (("I", "interpolation"), ("E", "extrapolation")):
        ds = [c["plain"]["nn"] - c["v3rat"]["nn"] for (c, tag) in cells.values()
              if tag == want and c["v3rat"]["ok"] and np.isfinite(c["plain"]["nn"])]
        wins = sum(x > 0 for x in ds)
        med, lo, hi = boot_ci(ds)
        print(f"  {lab}: {wins}/{len(ds)} cells win; paired bootstrap median improvement "
              f"{med:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  (CI>0: {lo > 0})", flush=True)
    unst = [k for k, (c, _) in cells.items() if not c["v3rat"]["ok"]]
    print(f"  v3rat unstable cells: {unst if unst else 'none'}")

    # ---------- Q3: size transfer at the V0 regime vs seeded mcsolve ----------
    print("\nQ3: M=8,10 at the V0 regime (0.8, 0.1, 0.02, 0.15) vs seeded mcsolve (exact refs).")
    gc, ka, gp, sg = V0REG
    for M, sds in ((8, [7, 8, 9]), (10, [7, 8])):
        for sd in sds:
            t0 = time.time()
            from scalable_engine import ScalableEngine
            w = draw_disorder(M, WC, sg, sd, "gauss")
            g = np.full(M, gc / np.sqrt(M))
            eng = ScalableEngine(M, NC, WC, w, g, kappa=ka, gdown=0.02, gphi=gp, Fsafe=6).compile()
            det = w - WC
            _, nref, zref, mcerr = mc_reference(M, NC, w, g, ntraj=200)
            m0 = m0_product(eng, 1.0)
            row = {}
            for name, c in {"plain": None, "v2": ClosureApply(lc2, eng, det),
                            "v3rat": ClosureApplyV3(lc3, eng, det, (gc, ka, gp),
                                                    xform=ratio_transform)}.items():
                _, traj = run_from(eng, m0, c)
                row[name] = metrics(eng, traj, nref, zref)
            print(f"  M={M} seed {sd} (MCerr {mcerr:.3f}): plain[{fmt(row['plain'])}]  "
                  f"v2[{fmt(row['v2'])}]  v3rat[{fmt(row['v3rat'])}]  ({time.time()-t0:.0f}s)", flush=True)

    np.savez_compressed("results/campaign_eval.npz",
                        q1=np.array(q1, dtype=object), q1b=np.array(q1b, dtype=object),
                        cells=np.array(cells, dtype=object))
    print("\nsaved results/campaign_eval.npz")
