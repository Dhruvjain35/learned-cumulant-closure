"""
Shared publication style for the manuscript figures (render-only; no science here).

One encoding per METHOD, used in every figure:
  exact reference      near-black, solid, no marker in traces / '*' in dot plots
  V7 (learned)         categorical slot 1 (blue),   solid,    circle
  second order         categorical slot 2 (orange), dashed,   square
  kappa4 = 0 (3rd)     categorical slot 3 (aqua),   dash-dot, triangle
  kappa5 = 0 (4th)     categorical slot 4 (yellow), dotted,   diamond
The four-slot palette passes the dataviz validator (light surface; worst adjacent CVD dE 9.1, normal-vision 22.9).
Aqua and yellow sit below 3:1 contrast, so every series also carries its own marker, line style and a direct label
or legend entry -- identity is never carried by color alone. Regimes are encoded by fill: low loss (POCKET) filled,
normal loss (CONTROL) hollow. Mechanism families use neutral ink: Aaz filled, a+z hollow.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
SURFACE = "#ffffff"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

METHOD = {
    "exact": dict(color=INK, ls="-", marker="*", label="exact reference"),
    "v7": dict(color=BLUE, ls="-", marker="o", label="learned closure (V7)"),
    "plain": dict(color=ORANGE, ls="--", marker="s", label="second order"),
    "t3": dict(color=AQUA, ls="-.", marker="^", label="third order (κ4 = 0)"),
    "t4": dict(color=YELLOW, ls=":", marker="D", label="fourth order (κ5 = 0)"),
}
REGIME = {"POCKET": dict(label="low loss (κ = 0.03)", filled=True),
          "CONTROL": dict(label="normal loss (κ = 0.15)", filled=False)}
DOUBLE = 7.0      # in, double-column width
SINGLE = 3.4


def apply():
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.2,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "lines.linewidth": 1.4, "lines.markersize": 5, "legend.frameon": False,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })


def mk(method, regime=None, **kw):
    """Scatter kwargs for a method, filled or hollow by regime."""
    m = METHOD[method]
    filled = True if regime is None else REGIME[regime]["filled"]
    d = dict(marker=m["marker"], edgecolors=m["color"], linewidths=0.9,
             facecolors=(m["color"] if filled else SURFACE), zorder=3)
    d.update(kw)
    return d


def panel(ax, letter, x=-0.14, y=1.04):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")


def save(fig, stem, outdir="figures"):
    import os
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, stem + ".pdf"), bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(os.path.join(outdir, stem + ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
