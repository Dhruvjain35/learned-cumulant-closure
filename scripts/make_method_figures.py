"""Method diagram and mechanism figure for the README, light and dark. No new simulation.

    python run.py scripts/make_method_figures.py

Inputs:  results/mechanism_m5.npz  (Result 42: pre-registered mechanism test at M = 5, six fresh draws per regime)
         portable_v7/models/v4/metadata.json, router.json  (network shape, moment types, routing rule)
Outputs: assets/img/method{,-dark}.png, assets/img/mechanism3d{,-dark}.png
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import PolyCollection

for f in ("/System/Library/Fonts/HelveticaNeue.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf"):
    try:
        font_manager.fontManager.addfont(f)
    except Exception:
        pass

THEMES = {
    "": dict(bg="#ffffff", fg="#0d0d0d", sub="#5b5b5b", box="#f6f7f9", edge="#d4d7dc", pane=(0.96, 0.96, 0.97, 1),
             v7="#1d6ff2", v7soft="#e6efff", plain="#e8590c", plainsoft="#fff1e8", exact="#0d0d0d", apz="#93b8f5"),
    "-dark": dict(bg="#0d1117", fg="#f0f3f6", sub="#9aa4ae", box="#161b22", edge="#30363d", pane=(0.08, 0.10, 0.13, 1),
                  v7="#4c9aff", v7soft="#132544", plain="#ff7b3a", plainsoft="#2a1609", exact="#f0f3f6", apz="#2f5ea8"),
}


def style(c):
    plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 13,
                         "figure.facecolor": c["bg"], "savefig.facecolor": c["bg"], "axes.facecolor": c["bg"],
                         "text.color": c["fg"], "axes.labelcolor": c["sub"], "xtick.color": c["sub"],
                         "ytick.color": c["sub"], "axes.edgecolor": c["edge"]})


def box(ax, x, y, w, h, title, body, c, fc=None, ec=None, tc=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012", fc=fc or c["box"],
                                ec=ec or c["edge"], lw=1.3))
    ax.text(x + 0.018, y + h - 0.035, title, fontsize=14, fontweight="bold", color=tc or c["fg"], va="top")
    ax.text(x + 0.018, y + h - 0.095, body, fontsize=11.8, color=c["sub"], va="top", linespacing=1.55)


def arrow(ax, p, q, c, col=None):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16, lw=1.6, color=col or c["sub"],
                                 shrinkA=0, shrinkB=0))


def method(sfx, c):
    meta = json.load(open("portable_v7/models/v4/metadata.json"))
    router = json.load(open("portable_v7/router.json"))
    lay = meta["types"][0]["layers"]
    arch = " → ".join([str(lay[0][0])] + [str(l[1]) for l in lay])
    fig = plt.figure(figsize=(13, 6.2))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    box(ax, 0.02, 0.60, 0.28, 0.33, "1  Track the cheap moments",
        "averages and pair correlations of the\ncavity field and every atom:\n⟨a⟩, ⟨σᵢ⟩, ⟨a†a⟩, ⟨a σᵢ⁺⟩, ⟨σᵢ⁺σⱼ⁻⟩ …\n141 equations at 5 atoms", c)
    box(ax, 0.36, 0.60, 0.28, 0.33, "2  The equations need more",
        "every equation for a pair correlation\ncontains three-body moments:\n⟨ABC⟩ = factorized part + κ₃\nκ₃ is the part nobody tracks", c)
    arrow(ax, (0.30, 0.765), (0.36, 0.765), c)
    box(ax, 0.70, 0.73, 0.28, 0.20, "Standard closure", "set κ₃ = 0\nburst peak +28.5% to +33.1% off", c,
        fc=c["plainsoft"], ec=c["plain"], tc=c["plain"])
    box(ax, 0.70, 0.47, 0.28, 0.20, "Learned closure", "predict κ₃ with a small network\nburst peak within 1.5% (30/30)", c,
        fc=c["v7soft"], ec=c["v7"], tc=c["v7"])
    arrow(ax, (0.64, 0.80), (0.70, 0.83), c, c["plain"])
    arrow(ax, (0.64, 0.72), (0.70, 0.58), c, c["v7"])
    # network detail
    box(ax, 0.02, 0.03, 0.40, 0.47, "3  What the network sees",
        "for each three-body moment, built from its own pieces:\n"
        "  the three single moments and three pair moments\n"
        "  the atoms' detunings, in units of the cavity loss\n"
        "  three system-wide averages (photons per atom,\n"
        "  mean inversion, mean coherence)\n"
        "  the regime: κ, γ↓ and γφ as ratios\n"
        "  short memory traces (low-loss branch only)", c)
    box(ax, 0.48, 0.03, 0.22, 0.38, "Shared network",
        f"{arch}, ReLU\none per moment type\n({meta['n_types']} types), shared by every\natom: relabel the atoms and\nthe output relabels with them", c,
        fc=c["v7soft"], ec=c["v7"], tc=c["v7"])
    box(ax, 0.76, 0.03, 0.22, 0.38, "Router (frozen)",
        f"κ ≤ {router['kappa_max']} and γφ ≤ {router['gamma_phi_max']}\n→ memory branch (V6)\notherwise\n→ memoryless branch (V4)\ndecided once at t = 0", c)
    arrow(ax, (0.42, 0.22), (0.48, 0.22), c)
    arrow(ax, (0.70, 0.22), (0.76, 0.22), c)
    arrow(ax, (0.59, 0.41), (0.80, 0.47), c, c["v7"])
    fig.savefig(f"assets/img/method{sfx}.png", dpi=150)
    plt.close(fig)


def mechanism(sfx, c):
    z = np.load("results/mechanism_m5.npz", allow_pickle=True)
    cells = list(z["cells"])
    cell = next(x for x in cells if x["regime"] == "POCKET" and x["seed"] == 500)
    ts = cell["ts"]
    sel = ts <= 12.0 + 1e-9
    ts = ts[sel]
    ex = np.real(cell["kap_ex_aaz"][sel])
    lv = np.real(cell["kap_v7tf_aaz"][sel])
    fig = plt.figure(figsize=(13.5, 6.6))
    ax = fig.add_axes([-0.05, -0.02, 0.66, 0.88], projection="3d")
    M = ex.shape[1]
    for i in range(M):
        ax.add_collection3d(PolyCollection([[(ts[0], 0)] + list(zip(ts, ex[:, i])) + [(ts[-1], 0)]],
                                           facecolors=[c["v7"]], edgecolors="none", alpha=0.10), zs=i, zdir="y")
        ax.plot(ts, np.full_like(ts, i), ex[:, i], color=c["exact"], lw=2.4)
        ax.plot(ts, np.full_like(ts, i), lv[:, i], color=c["v7"], lw=1.8, ls=(0, (4, 2)))
        ax.plot(ts, np.full_like(ts, i), np.zeros_like(ts), color=c["plain"], lw=1.3, alpha=0.9)
    ax.set_xlim(0, 12); ax.set_ylim(-0.5, M - 0.5)
    lo, hi = min(ex.min(), lv.min()), max(ex.max(), lv.max(), 0.05)
    ax.set_zlim(lo * 1.05, hi + 0.05)
    ax.set_yticks(range(M)); ax.set_yticklabels([f"atom {i + 1}" for i in range(M)], fontsize=10)
    ax.set_xlabel("time", labelpad=8); ax.set_zlabel("Re κ₃", labelpad=6)
    ax.view_init(elev=22, azim=-60)
    ax.set_box_aspect((2.1, 1.3, 0.8))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color(c["pane"]); axis._axinfo["grid"].update(color=c["edge"], linewidth=0.6)
        axis.line.set_color(c["edge"])
    ax.tick_params(pad=2, labelsize=10.5)
    fig.text(0.04, 0.93, "The term the standard closure throws away", fontsize=15, fontweight="bold", color=c["fg"])
    fig.text(0.04, 0.905, "κ₃ of the dominant channel ⟨a†a σᶻ⟩ for each atom, held-out draw: exact (solid), the learned\n"
                          "closure's prediction from the same moments (dashed), and the standard closure's zero (orange).",
             fontsize=12, color=c["sub"], va="top", linespacing=1.5)
    # influence split across the twelve fresh draws
    bx = fig.add_axes([0.66, 0.25, 0.30, 0.44])
    order = [x for x in cells if x["regime"] == "POCKET"] + [x for x in cells if x["regime"] == "CONTROL"]
    y = np.arange(len(order))[::-1] + np.array([0.5 if x["regime"] == "POCKET" else 0 for x in order])
    sa = np.array([x["share_aaz"] for x in order]); sp = np.array([x["share_apz"] for x in order])
    bx.barh(y, sa, color=c["v7"], height=0.72, label="⟨a†a σᶻ⟩  photon number × inversion")
    bx.barh(y, sp, left=sa, color=c["apz"], height=0.72, label="⟨a σ⁺ σᶻ⟩  field × polarization × inversion")
    bx.set_yticks([y[2] + 0.5, y[8] + 0.5]); bx.set_yticklabels(["low loss\n6 fresh draws", "normal loss\n6 fresh draws"])
    bx.set_xlim(0, 1); bx.set_xticks([0, 0.25, 0.5, 0.75, 1]); bx.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    for s in ("top", "right", "left"):
        bx.spines[s].set_visible(False)
    bx.tick_params(length=0)
    bx.set_xlabel("share of the correction's influence")
    lg = bx.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.2), frameon=False, fontsize=11, ncol=1)
    nabl = sum(int(x["abl_unstable"]["Aaz"]) + int(x["abl_unstable"]["a+z"]) for x in order if x["regime"] == "POCKET")
    bx.set_title(f"Which channels carry it\n(pre-registered test, 5 atoms, 12 fresh draws).\nRemoving either one at low "
                 f"loss destabilizes\nthe closure in {nabl} of 12 runs.", fontsize=11.8, color=c["sub"], loc="left", pad=12)
    fig.savefig(f"assets/img/mechanism3d{sfx}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    for sfx, c in THEMES.items():
        style(c)
        method(sfx, c)
        mechanism(sfx, c)
    print("wrote assets/img/method{,-dark}.png and mechanism3d{,-dark}.png")
