"""Campaign heatmaps: v3-RATIO NRMSE_n per cell (annotated), training hull outlined, unstable cells X,
cells where plain cumulant wins hatched. The honest map: accuracy AND the failure pocket on one figure."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# reclassified cells: D = in-distribution tuple/unseen seed, I = true interpolation, E = extrapolation
cells = np.load("results/campaign_cells_reclassified.npz", allow_pickle=True)["cells"].item()

MAPS = {
    "A": dict(x=[0.0, 0.05, 0.15, 0.25, 0.35, 0.4], y=[0.3, 0.4, 0.55, 0.7, 0.85, 1.0, 1.1],
              xl="disorder sigma", yl="collective coupling g", ttl="g x sigma  (kappa=0.15, gphi=0.02)",
              hull=([0.05, 0.3], [0.4, 1.0])),
    "B": dict(x=[0.01, 0.02, 0.05, 0.1, 0.2, 0.3], y=[0.03, 0.05, 0.1, 0.15, 0.3, 0.4],
              xl="dephasing gphi", yl="cavity loss kappa", ttl="kappa x gphi  (g=0.7, sigma=0.15)",
              hull=([0.02, 0.1], [0.05, 0.3])),
}

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
for ax, (mk, mm) in zip(axes, MAPS.items()):
    nx, ny = len(mm["x"]), len(mm["y"])
    V = np.full((ny, nx), np.nan)
    for iy, yv in enumerate(mm["y"]):
        for ix, xv in enumerate(mm["x"]):
            r, tag = cells[(mk, yv, xv)]
            V[iy, ix] = r["v3rat"]["nn"] if r["v3rat"]["ok"] else np.nan
    im = ax.imshow(V, origin="lower", aspect="auto", cmap="viridis_r", vmin=0, vmax=0.5)
    for iy, yv in enumerate(mm["y"]):
        for ix, xv in enumerate(mm["x"]):
            r, tag = cells[(mk, yv, xv)]
            if not r["v3rat"]["ok"]:
                ax.text(ix, iy, "X", ha="center", va="center", color="red", fontsize=15, weight="bold")
            else:
                lose = r["v3rat"]["nn"] >= r["plain"]["nn"]
                ax.text(ix, iy, f"{r['v3rat']['nn']:.2f}" + ("*" if lose else "") + f"\n{tag}",
                        ha="center", va="center",
                        color="white" if V[iy, ix] > 0.25 else "black", fontsize=7)
    # hull outline: rectangle around the grid values that fall inside the hull ranges
    hx, hy = mm["hull"]
    xi = [i for i, v in enumerate(mm["x"]) if hx[0] <= v <= hx[1]]
    yi = [i for i, v in enumerate(mm["y"]) if hy[0] <= v <= hy[1]]
    ax.add_patch(plt.Rectangle((min(xi) - 0.5, min(yi) - 0.5),
                               max(xi) - min(xi) + 1, max(yi) - min(yi) + 1,
                               fill=False, ec="orange", lw=2))
    ax.set_xticks(range(nx), [str(v) for v in mm["x"]])
    ax.set_yticks(range(ny), [str(v) for v in mm["y"]])
    ax.set_xlabel(mm["xl"]); ax.set_ylabel(mm["yl"]); ax.set_title(mm["ttl"], fontsize=10)
    fig.colorbar(im, ax=ax, label="v3-RATIO NRMSE (photon number)")
fig.suptitle("Learned closure across regimes, exact-referenced (M=5, held-out disorder). orange = hull; "
             "X = unstable; * = plain better; D/I/E = in-distribution / interpolation / extrapolation",
             fontsize=9)
fig.tight_layout()
fig.savefig("results/campaign_heatmaps.png", dpi=150)
print("saved results/campaign_heatmaps.png")
