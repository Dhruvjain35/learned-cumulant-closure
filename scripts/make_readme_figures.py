"""Render the README figures (light and dark) from committed results only. No simulation.

    python run.py scripts/make_readme_figures.py

Inputs:  results/baseline_burst_trace_M3_pocket_draw50.npz   (hero: one cell, every closure)
         results/heldout_param_burst.npz                      (scoreboard: the 30 pre-registered held-out cells)
         results/v8_accuracy_ladder.log, v8_accuracy_M16.log, v8_disordered_ladder.log, v8_disordered_m12.log
Outputs: assets/img/{hero,scoreboard,scaling}{,-dark}.png
"""
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for f in ("/System/Library/Fonts/HelveticaNeue.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf"):
    try:
        font_manager.fontManager.addfont(f)
    except Exception:
        pass
FAMILY = ["Helvetica Neue", "Arial", "DejaVu Sans"]

R, OUT = "results/", "assets/img/"
THEMES = {
    "": dict(bg="#ffffff", fg="#0d0d0d", sub="#5b5b5b", grid="#e9e9e9", exact="#0d0d0d",
             plain="#e8590c", t3="#8b5cf6", t4="#d6336c", v7="#1d6ff2", v7soft="#cfe0fd"),
    "-dark": dict(bg="#0d1117", fg="#f0f3f6", sub="#9aa4ae", grid="#21262d", exact="#f0f3f6",
                  plain="#ff7b3a", t3="#a78bfa", t4="#ff5c8a", v7="#4c9aff", v7soft="#1b2c4a"),
}


def style(c):
    plt.rcParams.update({
        "font.family": FAMILY, "font.size": 15, "axes.facecolor": c["bg"], "figure.facecolor": c["bg"],
        "savefig.facecolor": c["bg"], "axes.edgecolor": c["sub"], "axes.labelcolor": c["sub"],
        "xtick.color": c["sub"], "ytick.color": c["sub"], "text.color": c["fg"], "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": 1.0, "xtick.major.size": 0, "ytick.major.size": 0,
        "xtick.major.pad": 8, "ytick.major.pad": 8, "axes.grid": True, "grid.color": c["grid"],
        "grid.linewidth": 1.0, "axes.axisbelow": True, "axes.titlelocation": "left",
    })


def hero(sfx, c):
    d = np.load(R + "baseline_burst_trace_M3_pocket_draw50.npz", allow_pickle=True)
    fig, ax = plt.subplots(figsize=(12, 5.6))
    T = 12.0
    ex_t, ex_n = d["ts_exact"], d["n_exact"]
    pk = ex_n.max()

    def peak(k):
        n, t = d["n_" + k], d["ts_" + k]
        i = int(np.argmax(np.where(t < 6, n, -np.inf)))
        return t[i], n[i]

    ax.plot(ex_t, ex_n, color=c["exact"], lw=6, solid_capstyle="round", zorder=3, alpha=.9)
    ax.plot(d["ts_plain"], d["n_plain"], color=c["plain"], lw=2.6, ls=(0, (6, 3)), zorder=4)
    ax.plot(d["ts_t3"], d["n_t3"], color=c["t3"], lw=2.4, ls=(0, (1.5, 2.5)), zorder=4, dash_capstyle="round")
    t4t, t4n = d["ts_t4"], d["n_t4"]
    ax.plot(t4t, t4n, color=c["t4"], lw=2.4, ls=(0, (7, 2, 1.5, 2)), zorder=4)
    ax.plot(d["ts_v7"], d["n_v7"], color=c["v7"], lw=2.6, zorder=5)

    from matplotlib.lines import Line2D
    def pct(k):
        _, n = peak(k)
        return f"{100 * (n - pk) / pk:+.1f}%".replace("-", "\u2212")
    handles = [Line2D([], [], color=c["exact"], lw=6, alpha=.9),
               Line2D([], [], color=c["v7"], lw=2.6),
               Line2D([], [], color=c["plain"], lw=2.6, ls=(0, (6, 3))),
               Line2D([], [], color=c["t3"], lw=2.4, ls=(0, (1.5, 2.5))),
               Line2D([], [], color=c["t4"], lw=2.4, ls=(0, (7, 2, 1.5, 2)))]
    labels = ["exact answer", f"learned closure   {pct('v7')}", f"2nd-order closure   {pct('plain')}",
              f"3rd-order closure   {pct('t3')}", "4th-order closure   blows up"]
    lg = ax.legend(handles, labels, loc="upper right", frameon=False, fontsize=14.5, handlelength=2.6,
                   labelspacing=0.7, borderaxespad=0.2)
    for t, h in zip(lg.get_texts(), handles):
        t.set_color(h.get_color()); t.set_fontweight("bold")
    ax.text(0.15, 4.62, "burst peak error, same system, same starting state", color=c["sub"], fontsize=13)

    ax.set_xlim(0, T)
    ax.set_ylim(-0.15, 4.9)
    ax.set_xlabel("time")
    ax.set_ylabel("photons in the cavity")
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(OUT + f"hero{sfx}.png", dpi=160)
    plt.close(fig)


def scoreboard(sfx, c):
    rows = list(np.load(R + "heldout_param_burst.npz", allow_pickle=True)["rows"])
    rows.sort(key=lambda r: r["v7"]["abs"])
    v7 = np.array([100 * r["v7"]["abs"] for r in rows])
    pl = np.array([100 * r["plain"]["abs"] for r in rows])
    x = np.arange(1, len(rows) + 1)
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.vlines(x, v7, pl, color=c["grid"], lw=2.2, zorder=1)
    ax.scatter(x, pl, s=70, marker="s", color=c["plain"], zorder=3, label="2nd-order closure")
    ax.scatter(x, v7, s=80, color=c["v7"], zorder=3, label="learned closure")
    ax.set_yscale("log")
    ax.set_ylim(0.012, 80)
    ax.set_yticks([0.1, 1, 10])
    ax.set_yticklabels(["0.1%", "1%", "10%"])
    ax.set_xlim(0.2, len(rows) + 0.8)
    ax.set_xticks([1, 10, 20, 30])
    ax.set_xlabel("30 held-out test systems (sorted)")
    ax.set_ylabel("burst-peak error")
    ax.grid(axis="x", visible=False)
    ax.text(1, 52, f"2nd-order closure: {pl.min():.1f}–{pl.max():.1f}% off on every one",
            color=c["plain"], fontsize=15, fontweight="bold", va="center")
    ax.text(1, 4.2, f"learned closure: median {np.median(v7):.2f}%, worst {v7.max():.2f}%, "
                      f"better on {int((v7 < pl).sum())}/{len(rows)}",
            color=c["v7"], fontsize=15, fontweight="bold", va="center")
    fig.tight_layout()
    fig.savefig(OUT + f"scoreboard{sfx}.png", dpi=160)
    plt.close(fig)


def ladder_rows():
    pat = re.compile(r"^(CONTROL|POCKET)\s+(\d+)\s+(?:v\d\s+)?\|\s+([\d.]+)\s+(?:[\d.]+%\s+\|\s+)?([+-][\d.]+)\s+([+-][\d.]+)")
    exact, mc = [], []
    for name, dest in (("v8_accuracy_ladder.log", exact), ("v8_accuracy_M16.log", exact),
                       ("v8_disordered_ladder.log", mc), ("v8_disordered_m12.log", mc)):
        for line in open(R + name):
            m = pat.match(line.strip())
            if m:
                dest.append((m[1], int(m[2]), float(m[4]), float(m[5])))
    return exact, mc


def scaling(sfx, c):
    exact, mc = ladder_rows()
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for reg, mk in (("CONTROL", "o"), ("POCKET", "D")):
        e = sorted(r for r in exact if r[0] == reg)
        if not e:
            continue
        M = [r[1] for r in e]
        ax.plot(M, [100 * abs(r[3]) for r in e], marker="s", ms=8, color=c["plain"], lw=2, alpha=.9)
        ax.plot(M, [100 * abs(r[2]) for r in e], marker="o", ms=9, color=c["v7"], lw=2.4)
    for reg in ("CONTROL", "POCKET"):
        q = sorted(r for r in mc if r[0] == reg)
        ax.scatter([r[1] for r in q], [100 * abs(r[2]) for r in q], s=90, facecolor=c["bg"],
                   edgecolor=c["v7"], lw=2, zorder=4)
        ax.scatter([r[1] for r in q], [100 * abs(r[3]) for r in q], s=80, marker="s", facecolor=c["bg"],
                   edgecolor=c["plain"], lw=2, zorder=4)
    ax.set_xlabel("number of emitters (exact answer gets exponentially harder)")
    ax.set_ylabel("burst-peak error")
    ax.set_xticks([5, 8, 10, 12, 16])
    ax.set_xlim(4.2, 16.8)
    ax.set_ylim(0, 39)
    ax.set_yticks([0, 10, 20, 30])
    ax.set_yticklabels(["0%", "10%", "20%", "30%"])
    ax.grid(axis="x", visible=False)
    ax.text(4.5, 33.4, "2nd-order closure", color=c["plain"], fontsize=15, fontweight="bold")
    ax.text(13.1, 2.0, "learned closure", color=c["v7"], fontsize=15, fontweight="bold")
    ax.text(16.7, 36.3, "filled: exact reference (no disorder)   hollow: quantum trajectories (with disorder)",
            color=c["sub"], fontsize=11.5, ha="right")
    fig.tight_layout()
    fig.savefig(OUT + f"scaling{sfx}.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    for sfx, c in THEMES.items():
        style(c)
        hero(sfx, c)
        scoreboard(sfx, c)
        scaling(sfx, c)
    print("wrote assets/img/{hero,scoreboard,scaling}{,-dark}.png")
