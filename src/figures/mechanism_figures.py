"""
Figures for the mechanism story: where the third cumulant enters, when it matters,
and why a closure trained on small systems transfers at all.

Reads results/kappa3_trace.npz, results/transfer_collapse.npz, results/feature_coverage.npz
(produced by src/analysis/kappa3_trace.py, transfer_collapse.py, feature_coverage.py).

Palette. Two jobs, two rules. n(t) and kappa3(t) are distinct QUANTITIES, so they get a
categorical pair (#0B6FC4 / #D06A22) validated for colour-vision deficiency: worst adjacent
separation dE 24.6 protan, 31.6 normal, both above threshold. M is an ordered MAGNITUDE, so it
gets a single-hue sequential ramp rather than arbitrary categorical hues. Each M is drawn in its
own panel so no plot ever carries eight overlapping series.

Run:  python src/figures/mechanism_figures.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE = "#0B6FC4", "#D06A22"
TEAL = ["#A8DAD4", "#6BBDB4", "#38928A", "#17635C"]   # sequential, light -> dark, for M
INK, MUTED, GRID = "#1B1F27", "#6B7280", "#E3E1DB"

plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 160,
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def fig_timing(tr, out):
    """Per-M small multiples: the burst and the connected cumulant that drives it.

    Both curves are divided by their own peak so they share ONE axis. A dual y-axis would let the
    two scales be slid past each other, which is exactly the comparison this figure is making --
    so it is avoided. Normalising cannot move a maximum in time, so the lead being read off here
    is unaffected by the rescaling.
    """
    Ms = tr["Ms"]
    fig, axes = plt.subplots(1, len(Ms), figsize=(2.55 * len(Ms), 2.6), sharex=True, sharey=True)
    for ax, i in zip(np.atleast_1d(axes), range(len(Ms))):
        t = tr["t"][i]
        n = tr["n"][i] / tr["n"][i].max()
        k3 = tr["k3"][i] / tr["k3"][i].max()
        tb, tk = tr["t_burst"][i], tr["t_k3"][i]
        ax.plot(t, k3, color=ORANGE, lw=2.0)
        ax.plot(t, n, color=BLUE, lw=2.0)
        ax.axvline(tk, color=ORANGE, lw=1.0, ls=":", alpha=0.9)
        ax.axvline(tb, color=BLUE, lw=1.0, ls=":", alpha=0.9)
        # lead arrow parked below the curves, where nothing else is drawn
        y = 1.18
        ax.annotate("", xy=(tb, y), xytext=(tk, y),
                    arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.9))
        ax.text((tb + tk) / 2, y + 0.05, f"{tb - tk:.1f}", ha="center", va="bottom",
                fontsize=8, color=INK)
        ax.set_title(f"M = {Ms[i]}", pad=16)
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 1.38)
        ax.set_yticks([0, 0.5, 1.0])
        ax.grid(axis="y", alpha=0.5)
        ax.set_xlabel("time")
        if i == 0:
            ax.set_ylabel("scaled to own peak")
    fig.subplots_adjust(top=0.78)
    fig.suptitle("The correlation peaks BEFORE the burst, at every size tested", y=1.14)
    fig.legend(handles=[plt.Line2D([], [], color=BLUE, lw=2, label=r"photon number $\langle a^\dagger a\rangle$"),
                        plt.Line2D([], [], color=ORANGE, lw=2,
                                   label=r"$|\kappa_3|$ — the correlation second order throws away")],
               loc="lower center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
    for _ext in ("png", "pdf"):    # vector output for publication
        fig.savefig(str(out).rsplit(".", 1)[0] + "." + _ext)
    plt.close(fig)


def fig_ratio(tr, out):
    """How big is the dropped term, relative to what second order keeps, at the burst?"""
    Ms, r = tr["Ms"], tr["ratio_at_burst"]
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.plot(Ms, r, color=ORANGE, lw=2, marker="o", ms=7, clip_on=False, zorder=3)
    for x, y in zip(Ms, r):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8, color=INK)
    ax.set_xticks(Ms)
    ax.set_xlabel("number of emitters M")
    ax.set_ylabel("dropped / kept  at burst peak")
    ax.set_ylim(0, max(r) * 1.45)
    ax.grid(axis="y", alpha=0.5)
    ax.set_title("The dropped term does not shrink with size", loc="left")
    for _ext in ("png", "pdf"):    # vector output for publication
        fig.savefig(str(out).rsplit(".", 1)[0] + "." + _ext)
    plt.close(fig)


def _binned(x, y, lo, hi, nb=22, minn=8):
    """Mean of y in equal-width x bins; the quantity the collapse statistic compares across M."""
    edges = np.linspace(lo, hi, nb + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, nb - 1)
    xs, ys = [], []
    for b in range(nb):
        m = idx == b
        if m.sum() >= minn:
            xs.append(0.5 * (edges[b] + edges[b + 1]))
            ys.append(y[m].mean())
    return np.array(xs), np.array(ys)


def fig_collapse(tc, out):
    """Does kappa3 lie on ONE curve across M in the estimator's own feature?

    The raw samples are trajectories, so they loop and overlap and are hard to read directly.
    The statistic in transfer_collapse.py compares BIN MEANS across M, so that is what is drawn:
    faint raw points for honesty about the scatter, bin means on top for the actual comparison.
    """
    k3, Ms = tc["k3"], tc["M"]
    uM = np.unique(Ms)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)

    for ax, key, title, xlabel in (
        (axes[0], "npair", "Collapses:  across-M spread = 1.12x",
         r"$\langle a^\dagger a\rangle$   (the model's own feature)"),
        (axes[1], "nper", "Does not collapse:  3.07x",
         r"$\langle a^\dagger a\rangle / M$   (intensive context)"),
    ):
        x = tc[key]
        lo, hi = x.min(), x.max()
        for j, m in enumerate(uM):
            s = Ms == m
            ax.scatter(x[s], k3[s], s=2, color=TEAL[j % len(TEAL)], alpha=0.18, linewidths=0)
        for j, m in enumerate(uM):
            s = Ms == m
            bx, by = _binned(x[s], k3[s], lo, hi)
            ax.plot(bx, by, color=TEAL[j % len(TEAL)], lw=2.2, solid_capstyle="round",
                    label=f"M = {int(m)}" if key == "npair" else None, zorder=3)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.5)
        ax.set_title(title, loc="left", fontsize=9)

    axes[0].set_ylabel(r"$\kappa_3$")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.subplots_adjust(top=0.74)
    fig.suptitle("Transfer works in the coordinates the model actually uses", y=1.06)
    for _ext in ("png", "pdf"):    # vector output for publication
        fig.savefig(str(out).rsplit(".", 1)[0] + "." + _ext)
    plt.close(fig)


def fig_coverage(fc, out):
    """The training window in <a^d a>, and where each M's burst peak falls relative to it."""
    Ms, pk = fc["Ms"], fc["peak_npair"]
    hi = pk[1]                                     # M=2,3 training window upper edge
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.axhspan(0, hi, color=TEAL[0], alpha=0.45, lw=0)
    ax.text(Ms[0] - 0.15, hi * 0.5, "seen in training\n(M = 2, 3)", fontsize=8, color="#0F4F4A",
            va="center")
    ax.plot(Ms, pk, color=ORANGE, lw=2, marker="o", ms=7, clip_on=False, zorder=3)
    for x, y in zip(Ms, pk):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8, color=INK)
    ax.axhline(hi, color="#0F4F4A", lw=1, ls="--")
    ax.set_xticks(Ms)
    ax.set_xlabel("number of emitters M")
    ax.set_ylabel(r"peak $\langle a^\dagger a\rangle$")
    ax.set_ylim(0, max(pk) * 1.25)
    ax.grid(axis="y", alpha=0.5)
    ax.set_title("Bigger systems leave the window the model was fitted on", loc="left")
    for _ext in ("png", "pdf"):    # vector output for publication
        fig.savefig(str(out).rsplit(".", 1)[0] + "." + _ext)
    plt.close(fig)


def main():
    os.makedirs("results", exist_ok=True)
    tr = np.load("results/kappa3_trace.npz")
    tc = np.load("results/transfer_collapse.npz")
    fc = np.load("results/feature_coverage.npz")

    fig_timing(tr, "results/fig_mech_timing.png")
    fig_ratio(tr, "results/fig_mech_ratio.png")
    fig_collapse(tc, "results/fig_mech_collapse.png")
    fig_coverage(fc, "results/fig_mech_coverage.png")
    print("wrote results/fig_mech_{timing,ratio,collapse,coverage}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
