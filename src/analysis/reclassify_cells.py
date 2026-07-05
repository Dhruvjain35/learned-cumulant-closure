"""
Claim-class correction (stop-sign from review): a heatmap cell whose regime tuple (gc,ka,gp,sg) is ON the
training grid is NOT interpolation evidence even with a held-out disorder seed -- it is in-distribution
generalization (class D). Reclassify every campaign cell into:
  D  in-distribution: regime tuple in the training grid; disorder seed unseen
  I  interpolation:   tuple inside the hull but NOT on the training grid; seed unseen
  E  extrapolation:   at least one parameter outside the hull; seed unseen
and recompute per-class wins + paired bootstrap CIs from the saved campaign cells. No retraining, no rerun.
"""
import numpy as np
from campaign_eval import in_hull, boot_ci

TR_GC, TR_KA, TR_GP, TR_SG = {0.4, 0.7, 1.0}, {0.05, 0.15, 0.3}, {0.02, 0.1}, {0.05, 0.15, 0.3}


def on_training_grid(gc, ka, gp, sg):
    return gc in TR_GC and ka in TR_KA and gp in TR_GP and sg in TR_SG


def klass(gc, ka, gp, sg):
    if on_training_grid(gc, ka, gp, sg):
        return "D"
    return "I" if in_hull(gc, ka, gp, sg) else "E"


if __name__ == "__main__":
    cells = np.load("results/campaign_eval.npz", allow_pickle=True)["cells"].item()
    reclassed = {}
    for key, (r, _old) in cells.items():
        mk = key[0]
        if mk == "A":
            gc, sg, ka, gp = key[1], key[2], 0.15, 0.02
        else:
            ka, gp, gc, sg = key[1], key[2], 0.7, 0.15
        reclassed[key] = (r, klass(gc, ka, gp, sg))

    print("corrected claim classes (D = in-distribution tuple/unseen seed; I = unseen tuple in hull; E = outside hull):")
    for want, lab in (("D", "in-distribution"), ("I", "interpolation"), ("E", "extrapolation")):
        ds, unstable = [], 0
        for (r, tag) in reclassed.values():
            if tag != want:
                continue
            if not r["v3rat"]["ok"]:
                unstable += 1
                continue
            ds.append(r["plain"]["nn"] - r["v3rat"]["nn"])
        wins = sum(x > 0 for x in ds)
        med, lo, hi = boot_ci(ds)
        print(f"  {lab:>16s}: {len(ds)} stable cells (+{unstable} unstable), {wins}/{len(ds)} win; "
              f"paired bootstrap median improvement {med:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  (CI>0: {lo > 0})")
    np.savez_compressed("results/campaign_cells_reclassified.npz",
                        cells=np.array(reclassed, dtype=object))
    print("saved results/campaign_cells_reclassified.npz")
