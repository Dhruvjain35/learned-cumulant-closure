"""The locked ladder figure (PROTOCOL pre-final lock, item 4): columns MF -> exact, rows = regions;
each cell = stability% + median conditional NRMSE; blowups marked; MF annotated as physics; T3 cells
carry their M<=4 annotation; V7's column is DERIVED from its bit-identical branches (v6 in-box, v4
elsewhere), never re-run. One seed set: 32/33/34."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

bl = np.load("results/baselines_final_s32_33_34.npz", allow_pickle=True)["res"].item()
pl = np.load("results/plain_final_s32_33_34.npz", allow_pickle=True)["res"].item()
pf = np.load("results/plain_farm_s32_33_34.npz", allow_pickle=True)["res"].item()

REGIONS = ["in-box", "seam+resid", "sanity", "far-M"]
COLS = ["MF", "plain 2nd", "L21", "L27", "Q21", "Q27", "T3*", "V7", "exact"]
# T3 final numbers (from t3_eval_final log; deep corner = in-box cell at M=3/4): med of stable, 1 blowup
T3 = {"in-box": (5 / 6, 0.375), "seam+resid": None, "sanity": (1.0, 0.460), "far-M": None}
# ('sanity' row for T3 = its 'normal' cells at (0.15, 0.02), M=3/4 -- annotated)


def sel(model, region):
    if region == "in-box":
        ks = [k for k in bl if k[0].startswith("in:") and k[-1] == model]
    elif region == "seam+resid":
        ks = [k for k in bl if k[0].startswith("out:") and k[-1] == model]
    elif region == "sanity":
        ks = [k for k in bl if k[0] == "sanity" and k[-1] == model]
    else:
        ks = [k for k in bl if k[0] == "farM" and k[-1] == model]
    st = np.mean([bl[k][0] for k in ks])
    vals = [bl[k][1] for k in ks if bl[k][0]]
    return st, (np.median(vals) if vals else np.nan)


def sel_plain(region):
    if region == "far-M":
        st = np.mean([pf[k][0] for k in pf])
        vals = [pf[k][1] for k in pf if pf[k][0]]
        return st, np.median(vals)
    if region == "in-box":
        ks = [k for k in pl if str(k[0]).startswith("in:")]
    elif region == "seam+resid":
        ks = [k for k in pl if str(k[0]).startswith("out:")]
    else:
        ks = [k for k in pl if k[0] == "sanity"]
    st = np.mean([pl[k][0] for k in ks])
    vals = [pl[k][1] for k in ks if pl[k][0]]
    return st, np.median(vals)


grid = np.full((len(REGIONS), len(COLS)), np.nan)
text = [["" for _ in COLS] for _ in REGIONS]
for ri, reg in enumerate(REGIONS):
    for ci, col in enumerate(COLS):
        if col == "MF":
            text[ri][ci] = "frozen\n(no seed)"
        elif col == "exact":
            text[ri][ci] = "0\n(ref)" if reg != "far-M" else "MC err\n~0.10-0.12"
        elif col == "plain 2nd":
            st, md = sel_plain(reg)
            grid[ri][ci] = md
            text[ri][ci] = f"{100*st:.0f}%\n{md:.3f}"
        elif col == "T3*":
            if T3[reg] is None:
                text[ri][ci] = "n/a"
            else:
                st, md = T3[reg]
                grid[ri][ci] = md
                text[ri][ci] = f"{100*st:.0f}%\n{md:.3f}"
        elif col == "V7":
            src = "v6" if reg == "in-box" else "v4"
            st, md = sel(src, reg)
            grid[ri][ci] = md
            text[ri][ci] = f"{100*st:.0f}%\n{md:.3f}\n[={src}]"
        else:
            st, md = sel(col.lower(), reg)
            grid[ri][ci] = md
            text[ri][ci] = f"{100*st:.0f}%\n{md:.3f}"

fig, ax = plt.subplots(figsize=(13, 4.6))
im = ax.imshow(grid, aspect="auto", cmap="viridis_r", vmin=0, vmax=0.45)
for ri in range(len(REGIONS)):
    for ci in range(len(COLS)):
        v = grid[ri][ci]
        ax.text(ci, ri, text[ri][ci], ha="center", va="center", fontsize=7.5,
                color=("white" if (not np.isnan(v) and v > 0.22) else "black"))
ax.set_xticks(range(len(COLS)), COLS)
ax.set_yticks(range(len(REGIONS)), REGIONS)
ax.set_title("Closure-structure ladder, untouched seeds 32/33/34 -- cell: stability% / median cond. NRMSE\n"
             "MF frozen for these ICs (physics); T3* = tested kappa4=0 truncation, M<=4 cells only "
             "(in-box = deep corner, sanity = its normal cells; 1 blowup of 6 in-box); V7 = routed branch "
             "(bit-identical)", fontsize=9)
fig.colorbar(im, ax=ax, label="median conditional NRMSE")
fig.tight_layout()
fig.savefig("results/ladder.png", dpi=150)
print("saved results/ladder.png")
