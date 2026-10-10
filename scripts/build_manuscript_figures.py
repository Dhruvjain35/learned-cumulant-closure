"""
Build the five main manuscript figures from COMMITTED artifacts only. Render-only: no exact solve, no closure
rollout, no network evaluation. Run from the repo root:  python run.py scripts/build_manuscript_figures.py
(or `make figures`). Writes figures/fig{1..5}_*.pdf, .eps and .png. Panel -> artifact map: docs/FIGURE_PROVENANCE.md.

Logs are parsed with strict patterns and the expected row counts are asserted, so a changed or truncated log
fails loudly instead of plotting the wrong numbers.
"""
import re
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Arc, FancyArrowPatch
from matplotlib.lines import Line2D

import pubstyle as S

S.apply()
R = "results/"
REGS = ["POCKET", "CONTROL"]


# ------------------------------------------------------------------ loaders (committed artifacts)
def burst_pred():
    d = np.load(R + "burst_pred_exact.npz", allow_pickle=True)["data"].item()
    out = {}
    for (reg, M, sd), (obs, _) in d.items():
        ref = obs["ref"]["npeak"]
        out[(reg, M, sd)] = dict(plain=obs["plain"]["npeak"] / ref - 1, v7=obs["v7"]["npeak"] / ref - 1)
    assert len(out) == 24
    return out


def parse_table(path, pattern, n):
    rows = [m.groups() for m in (re.match(pattern, l) for l in open(path)) if m]
    assert len(rows) == n, (path, len(rows), n)
    return rows


def piqs_ladder():
    pat = r"^(CONTROL|POCKET)\s+(\d+)\s+v[46]\s+\|\s+([\d.]+)\s+([+-][\d.]+)\s+([+-][\d.]+)"
    rows = parse_table(R + "v8_accuracy_ladder.log", pat, 8)
    pat16 = r"^(CONTROL|POCKET)\s+(16)\s+\|\s+([\d.]+)\s+([+-][\d.]+)\s+([+-][\d.]+)"
    rows += parse_table(R + "v8_accuracy_M16.log", pat16, 2)
    out = {}
    for reg, M, n, v7, pl in rows:
        out[(reg, int(M))] = dict(v7=float(v7), plain=float(pl))
    return out


def disordered_m12():
    pat = r"^(POCKET|CONTROL)\s+12\s+\|\s+([\d.]+)\s+([\d.]+)%\s+\|\s+([+-][\d.]+)\s+([+-][\d.]+)"
    return {reg: dict(npeak=float(n), mc=float(mc) / 100, v7=float(v7), plain=float(pl))
            for reg, n, mc, v7, pl in parse_table(R + "v8_disordered_m12.log", pat, 2)}


def peak_vs_violation():
    pat = r"^\s+(\d+)\s+v6\s+\|\s+([\d.]+)\s+([\d.]+)"
    return [(int(M), float(tp), float(tv)) for M, tp, tv in parse_table(R + "v8_peak_vs_violation.log", pat, 3)]


def m12_timing():
    txt = open(R + "m12_violation_timing.log").read()
    out = {}
    for reg in REGS:
        blk = txt.split(reg + " (")[1]
        tp = float(re.search(r"burst peak \(max over t <= 10\):\s+n = [\d.]+ at t = ([\d.]+)", blk).group(1))
        none = re.search(r"first physicality violation:\s+none to t = 40", blk.split("max per-site")[0]) is not None
        out[reg] = dict(tpeak=tp, violation=None if none else "see log")
    assert all(v["violation"] is None for v in out.values())
    return out


def mech_m4():
    """Result 25 aggregate rows (committed results/mechanism.log): D share and mean finite frac_lost per family."""
    out, reg = {}, None
    for l in open(R + "mechanism.log"):
        m = re.match(r"^\[(POCKET|CONTROL) M4\]", l)
        if m:
            reg = m.group(1)
        m = re.match(r"^\s+(Aaz|a\+z)\s+\|\s+[\d.]+\s+([\d.]+)\s+([+-][\d.]+)", l)
        if m and reg:
            out[(reg, m.group(1))] = dict(share=float(m.group(2)), loss=float(m.group(3)))
    assert len(out) == 4
    return out


# ------------------------------------------------------------------ helpers
def box(ax, x, y, w, h, text, fc="#f4f3ef", ec=S.INK2, fs=9, bold=False, color=S.INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.02", fc=fc, ec=ec, lw=0.7,
                                clip_on=False))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=S.fs(fs), color=color,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x0, y0, x1, y1, color=S.INK2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, lw=0.8, color=color))


def jitter(n, w=0.18, seed=0):
    return np.linspace(-w, w, n) if n > 1 else np.zeros(1)


# ------------------------------------------------------------------ Figure 1: problem and method
def fig1():
    from training_support import TRAIN
    from matplotlib.patches import Patch
    fig = plt.figure(figsize=(S.DOUBLE + 0.4, 5.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1.12], hspace=0.32, wspace=0.3)
    # (a) model schematic
    ax = fig.add_subplot(gs[0, 0]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); S.panel(ax, "a", x=-0.02, y=1.0)
    for xm, th in ((0.2, (300, 60)), (0.8, (120, 240))):
        ax.add_patch(Arc((xm, 0.66), 0.14, 0.44, theta1=th[0], theta2=th[1], lw=2.2, color=S.INK2))
    xs = np.linspace(0.32, 0.68, 5)
    ys = 0.66 + 0.055 * np.array([1, -1, 0.5, -0.6, 0.9])
    for x, y in zip(xs, ys):
        ax.add_patch(Circle((x, y), 0.024, fc=S.BLUE, ec=S.INK, lw=0.6))
    ax.text(0.5, 0.97, "cavity mode $a$, $M$ emitters, disorder σ", ha="center", va="top", fontsize=S.fs(9))
    arrow(ax, 0.86, 0.66, 0.96, 0.66); ax.text(0.91, 0.70, "κ", ha="center", fontsize=S.fs(9))
    ax.annotate("", xy=(xs[1], ys[1] - 0.12), xytext=(xs[1], ys[1] - 0.033),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color=S.INK2))
    ax.text(xs[1] + 0.025, ys[1] - 0.15, "γ↓", fontsize=S.fs(9))
    ax.text(xs[3] - 0.01, ys[3] - 0.15, "γφ", fontsize=S.fs(9))
    ax.text(0.5, 0.31, "$g_i = g_c/\\sqrt{M}$\n"
            "initial state: vacuum, all emitters excited\n"
            "observable: $F_{\\rm peak} = \\kappa\\,\\max_t\\langle a^\\dagger a\\rangle$",
            ha="center", va="top", fontsize=S.fs(9), linespacing=1.55)
    # (b) closure problem
    ax = fig.add_subplot(gs[0, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); S.panel(ax, "b", x=-0.02, y=1.0)
    box(ax, 0.0, 0.70, 0.44, 0.2, "tracked moments\n(orders 1 and 2)")
    box(ax, 0.56, 0.70, 0.44, 0.2, "exact equations\nneed order 3")
    arrow(ax, 0.44, 0.80, 0.56, 0.80)
    box(ax, 0.0, 0.37, 0.44, 0.2, "second order:\nκ3 = 0 (≈ +29%)", fc="#fbeee8")
    box(ax, 0.56, 0.37, 0.44, 0.2, "learned closure:\npredicts κ3", fc="#eaf2fb")
    arrow(ax, 0.66, 0.70, 0.30, 0.57); arrow(ax, 0.78, 0.70, 0.78, 0.57)
    box(ax, 0.0, 0.04, 1.0, 0.2, "order-3 moment = factorization + κ3\n(same second-order engine)")
    arrow(ax, 0.22, 0.37, 0.34, 0.24); arrow(ax, 0.78, 0.37, 0.66, 0.24)
    # (c) architecture
    ax = fig.add_subplot(gs[1, 0]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); S.panel(ax, "c", x=-0.02, y=1.0)
    feats = ["own-factor moments", "detunings Δ/κ", "intensive context", "regime ratios", "memory traces*"]
    h, gap, y0 = 0.12, 0.055, 0.9
    for t in feats:
        y0 -= h
        box(ax, 0.0, y0, 0.5, h, t, fc="#f7f6f2")
        arrow(ax, 0.5, y0 + h / 2, 0.58, 0.5)
        y0 -= gap
    box(ax, 0.58, 0.38, 0.29, 0.24, "shared MLP\n2×128 ReLU", fc="#eaf2fb")
    arrow(ax, 0.87, 0.5, 0.9, 0.5)
    box(ax, 0.9, 0.42, 0.1, 0.16, "κ3")
    ax.text(0.0, 0.0, "*memory-aware branch only", ha="left", va="bottom", fontsize=S.fs(9), color=S.INK2)
    # (d) router + training support in the (kappa, gamma_phi) plane
    ax = fig.add_subplot(gs[1, 1]); S.panel(ax, "d", x=-0.2, y=1.0)
    router = json.load(open("portable_v7/router.json"))
    kmax, gmax = router["kappa_max"], router["gamma_phi_max"]
    ax.add_patch(Rectangle((0.018, 0.0), kmax - 0.018, gmax, fc="#eaf2fb", ec=S.BLUE, lw=0.9, zorder=0))
    kg = sorted({(t[1], t[2]) for t in TRAIN})
    ax.scatter([k for k, g in kg], [g for k, g in kg], marker="o", s=22, c=S.INK2, zorder=3)
    cells = json.load(open("results/heldout_param_cells.json"))["cells"]
    ax.scatter([c["kappa"] for c in cells], [c["gamma_phi"] for c in cells], marker="o", s=S.MS, facecolors=S.light(S.BLUE),
               edgecolors=S.BLUE, linewidths=0.9, zorder=4)
    ax.scatter([0.03, 0.15], [0.02, 0.02], marker="*", s=120, c=S.INK, zorder=5)
    ax.set_xscale("log"); ax.set_xlim(0.018, 0.5); ax.set_ylim(0, 0.13)
    ax.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5]); ax.set_xticklabels(["0.02", "0.05", "0.1", "0.2", "0.5"])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("cavity loss κ"); ax.set_ylabel("pure dephasing γφ")
    ax.legend(handles=[Patch(fc="#eaf2fb", ec=S.BLUE, label="memory-aware region"),
                       Line2D([], [], ls="", marker="o", mfc=S.INK2, mec=S.INK2, ms=5, label="training"),
                       Line2D([], [], ls="", marker="o", mfc=S.light(S.BLUE), mec=S.BLUE, ms=6, label="held-out (Result 40)"),
                       Line2D([], [], ls="", marker="*", color=S.INK, ms=9, label="headline cells")],
              loc="upper center", bbox_to_anchor=(0.45, -0.24), ncol=2, fontsize=S.fs(9), handletextpad=0.4,
              columnspacing=1.0)
    S.save(fig, "fig1_problem_method")


# ------------------------------------------------------------------ Figure 2: accuracy and parameter transfer
def fig2():
    # Three data panels in one row. The summary by support class is Table I of the manuscript, not a figure inset.
    fig = plt.figure(figsize=(S.DOUBLE, 3.25))
    gs = fig.add_gridspec(1, 3, wspace=0.42, width_ratios=[1.05, 1, 1])
    # (a) representative trace (canonical cell: M=3 low loss draw 50, Results 24/27)
    d = np.load(R + "baseline_burst_trace_M3_pocket_draw50.npz", allow_pickle=True)
    ax = fig.add_subplot(gs[0, 0]); S.panel(ax, "a", x=-0.3)
    for key, tk, nk in (("exact", "ts_exact", "n_exact"), ("plain", "ts_plain", "n_plain"), ("v7", "ts_v7", "n_v7")):
        m = S.METHOD[key]
        ax.plot(d[tk], d[nk], color=m["color"], ls=m["ls"], lw=1.6 if key != "exact" else 2.2,
                label=m["label"], zorder=3 if key == "v7" else 2)
    ax.set_xlim(0, 15); ax.set_xlabel("time t"); ax.set_ylabel("⟨a†a⟩")
    # (b) original exact M<=5 benchmark
    bp = burst_pred()
    ax = fig.add_subplot(gs[0, 1]); S.panel(ax, "b", x=-0.3)
    for meth in ("plain", "v7"):
        for reg in REGS:
            off = -0.09 if reg == "POCKET" else 0.09
            for M in (2, 3, 4, 5):
                vals = [abs(bp[(reg, M, s)][meth]) for s in (50, 51, 52)]
                ax.scatter(np.full(3, M + off), vals, s=S.MS, **S.mk(meth, reg))
    ax.set_yscale("log"); ax.set_ylim(5e-5, 1); ax.set_xticks([2, 3, 4, 5]); ax.set_xlim(1.6, 5.5)
    ax.set_xlabel("emitters M"); ax.set_ylabel("|burst-peak error|")
    # (c) Result 40 -- all 30 cells
    hp = np.load(R + "heldout_param_burst.npz", allow_pickle=True)
    rows = hp["rows"]
    ax = fig.add_subplot(gs[0, 2]); S.panel(ax, "c", x=-0.3)
    classes = [("B", "interp."), ("C", "1-axis"), ("D", "2-axis")]
    for i, (cls, _) in enumerate(classes):
        rs = [r for r in rows if r["cls"] == cls]
        j = jitter(len(rs), 0.13)
        for meth, dx in (("plain", -0.18), ("v7", 0.18)):
            vals = np.array([r[meth]["abs"] for r in rs])
            ax.scatter(i + dx + j * 0.6, vals, s=22, **S.mk(meth, None))
            med, q1, q3 = np.median(vals), np.percentile(vals, 25), np.percentile(vals, 75)
            ax.plot([i + dx - 0.13, i + dx + 0.13], [med, med], color=S.INK, lw=1.4, zorder=4)
            ax.plot([i + dx, i + dx], [q1, q3], color=S.INK, lw=0.8, zorder=4)
    ax.set_yscale("log"); ax.set_ylim(5e-5, 1); ax.set_xticks(range(3)); ax.set_xticklabels([c[1] for c in classes])
    ax.set_xlim(-0.6, 2.6)
    ax.set_xlabel("support class"); ax.set_ylabel("|burst-peak error|")
    fig.legend(handles=[Line2D([], [], color=S.INK, lw=2.2, label="exact reference"),
                        Line2D([], [], color=S.ORANGE, ls="--", marker="s", ms=5, label="second order"),
                        Line2D([], [], color=S.BLUE, marker="o", ms=5, label="learned closure (V7)"),
                        Line2D([], [], ls="", marker="o", mfc=S.INK2, mec=S.INK2, label="low loss"),
                        Line2D([], [], ls="", marker="o", mfc=S.light(S.INK2), mec=S.INK2, label="normal loss"),
                        Line2D([], [], color=S.INK, lw=1.4, label="median, IQR")],
               loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=6, fontsize=S.fs(9), handlelength=1.6,
               columnspacing=1.0, handletextpad=0.4)
    S.save(fig, "fig2_accuracy_parameter_transfer")


# ------------------------------------------------------------------ Figure 3: size transfer and validity envelope
def fig3():
    lad, dm = piqs_ladder(), disordered_m12()
    fig = plt.figure(figsize=(S.DOUBLE, 6.7))
    gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.62, width_ratios=[1.1, 1])
    # (a) signed error vs M (zero-disorder exact permutation-invariant ladder), measured points only
    ax = fig.add_subplot(gs[0, 0]); S.panel(ax, "a", x=-0.2)
    Ms = [5, 8, 10, 12, 16]
    for reg in REGS:
        for meth in ("plain", "v7"):
            y = [100 * lad[(reg, M)][meth] for M in Ms]
            m = S.METHOD[meth]
            ax.plot(Ms, y, color=m["color"], ls=m["ls"], lw=1.0, zorder=2)
            ax.scatter(Ms, y, s=S.MS, **S.mk(meth, reg))
    for reg in REGS:   # disordered M = 12, one draw, quantum trajectories (MC error bar)
        for meth in ("plain", "v7"):
            v = 100 * dm[reg][meth]
            ax.errorbar(12.55 if reg == "POCKET" else 12.85, v, yerr=100 * dm[reg]["mc"], fmt="none",
                        ecolor=S.METHOD[meth]["color"], elinewidth=0.9, capsize=2, zorder=2)
            ax.scatter(12.55 if reg == "POCKET" else 12.85, v, s=26, marker="x", c=S.METHOD[meth]["color"], zorder=4)
    ax.axhline(0, color=S.MUTED, lw=0.6)
    ax.set_xticks(Ms); ax.set_xlabel("emitters M (trained at M ≤ 5)"); ax.set_ylabel("signed burst-peak error (%)")
    ax.set_title("size transfer: measured points only", fontsize=S.fs(8))
    ax.text(16.7, 100 * lad[("CONTROL", 16)]["plain"], "second order", color=S.ORANGE, fontsize=S.fs(7), va="center")
    ax.text(16.7, 100 * lad[("POCKET", 16)]["v7"], "V7", color=S.BLUE, fontsize=S.fs(7), va="center")
    ax.set_xlim(4, 18.2)
    ax.legend(handles=[Line2D([], [], ls="", marker="o", mfc=S.INK2, mec=S.INK2, label="σ = 0, exact; low loss (dark)"),
                       Line2D([], [], ls="", marker="o", mfc=S.light(S.INK2), mec=S.INK2, label="σ = 0, exact; normal loss (light)"),
                       Line2D([], [], ls="", marker="x", color=S.INK2, label="σ = 0.15, trajectories, one draw (±MC)")],
              loc="center left", fontsize=S.fs(6.2), bbox_to_anchor=(0.0, 0.58))
    # (b) M = 12: disordered trajectories vs zero-disorder exact
    ax = fig.add_subplot(gs[0, 1]); S.panel(ax, "b", x=-0.2)
    for i, reg in enumerate(REGS):
        for meth, dx in (("plain", -0.12), ("v7", 0.12)):
            a = 100 * lad[(reg, 12)][meth]; b = 100 * dm[reg][meth]
            ax.scatter(i + dx - 0.05, a, s=S.MS, **S.mk(meth, None))
            ax.errorbar(i + dx + 0.05, b, yerr=100 * dm[reg]["mc"], fmt="none", ecolor=S.METHOD[meth]["color"], capsize=2)
            ax.scatter(i + dx + 0.05, b, s=30, marker="x", c=S.METHOD[meth]["color"], zorder=4)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["low loss", "normal loss"]); ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("signed error at M = 12 (%)"); ax.set_ylim(0, 34)
    ax.set_title("M = 12: σ = 0 exact vs σ = 0.15 trajectories", fontsize=S.fs(8))
    ax.legend(handles=[Line2D([], [], ls="", marker="s", mfc=S.INK2, mec=S.INK2, label="σ = 0, exact permutation-invariant"),
                       Line2D([], [], ls="", marker="x", color=S.INK2, label="σ = 0.15, 400 trajectories, one draw")],
              loc="upper center", bbox_to_anchor=(0.45, -0.18), fontsize=S.fs(6.2))
    # (c) full-trajectory physicality map (V7), t <= 40
    ax = fig.add_subplot(gs[1, 0]); S.panel(ax, "c", x=-0.3)
    pos = np.load(R + "positivity_audit.npz", allow_pickle=True)["rows"]
    s0 = np.load(R + "v8_step0_sigma0.npz", allow_pickle=True)["rows"][0]
    Mcols = [5, 8, 10, 12, 16, 20]
    rowsdef = []
    # low loss, sigma = 0.15: Result 30 (seeds 7-9) + Result 41 (seed 50, M = 12)
    r30 = {M: [(r["under"] <= 1e-6 and r["over"] <= 1e-6 and r["n_min"] >= -1e-6) for r in pos
               if r["closure"] == "V7" and r["M"] == M] for M in Mcols}
    r30[12] = r30[12] + [True]
    rowsdef.append(("low loss, σ = 0.15", r30))
    rowsdef.append(("low loss, σ = 0", {M: [s0[("POCKET", M)]["V7"]["physical"]] for M in Mcols}))
    rowsdef.append(("normal loss, σ = 0", {M: [s0[("CONTROL", M)]["V7"]["physical"]] for M in Mcols}))
    rowsdef.append(("normal loss, σ = 0.15", {M: ([True] if M == 12 else []) for M in Mcols}))
    for i, (name, dd) in enumerate(rowsdef):
        y = len(rowsdef) - 1 - i
        for j, M in enumerate(Mcols):
            v = dd[M]
            if not v:
                sym, col, txt = "", S.MUTED, "–"
            elif all(v):
                sym, col, txt = "o", S.BLUE, f"{sum(v)}/{len(v)}"
            elif not any(v):
                sym, col, txt = "X", S.ORANGE, f"{sum(v)}/{len(v)}"
            else:
                sym, col, txt = "o", S.YELLOW, f"{sum(v)}/{len(v)}"
            if sym:
                filled = all(v)
                ax.scatter(j, y + 0.12, marker=sym, s=70, facecolors=col if (filled or sym == "X") else S.light(col),
                           edgecolors=col, linewidths=1.2, zorder=3)
            ax.text(j, y - 0.27, txt, ha="center", va="center", fontsize=S.fs(6.6), color=S.INK2)
    ax.set_yticks(range(len(rowsdef))); ax.set_yticklabels([r[0] for r in rowsdef][::-1], fontsize=S.fs(7))
    ax.set_xticks(range(len(Mcols))); ax.set_xticklabels([str(M) for M in Mcols]); ax.set_xlabel("emitters M")
    ax.set_xlim(-0.5, len(Mcols) - 0.5); ax.set_ylim(-0.6, len(rowsdef) - 0.4)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.set_title("V7 physicality over t ≤ 40\n(physical / tested)", fontsize=S.fs(8))
    ax.legend(handles=[Line2D([], [], ls="", marker="o", mfc=S.BLUE, mec=S.BLUE, label="all tested draws physical"),
                       Line2D([], [], ls="", marker="o", mfc=S.light(S.YELLOW), mec=S.YELLOW, mew=1.2, label="draw-dependent"),
                       Line2D([], [], ls="", marker="X", mfc=S.ORANGE, mec=S.ORANGE, label="leaves physical set"),
                       Line2D([], [], ls="", marker="", label="– not measured")],
              loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2, fontsize=S.fs(6.2), handletextpad=0.2, columnspacing=0.8)
    # (d) burst time vs first violation time
    ax = fig.add_subplot(gs[1, 1]); S.panel(ax, "d", x=-0.55)
    pv, mt = peak_vs_violation(), m12_timing()
    rows = [(f"low loss, M = {M}\nσ = 0", tp, tv) for M, tp, tv in pv]
    rows += [(f"{'low' if r == 'POCKET' else 'normal'} loss, M = 12\nσ = 0.15, one draw", mt[r]["tpeak"], None) for r in REGS]
    for i, (name, tp, tv) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.scatter(tp, y, marker="o", s=26, c=S.BLUE, zorder=3)
        if tv is not None:
            ax.plot([tp, tv], [y, y], color=S.MUTED, lw=0.8)
            ax.scatter(tv, y, marker="X", s=34, c=S.ORANGE, zorder=3)
        else:
            ax.annotate("", xy=(40, y), xytext=(tp, y), arrowprops=dict(arrowstyle="-|>", color=S.MUTED, lw=0.8))
            ax.text(22, y + 0.18, "no violation to t = 40", fontsize=S.fs(6.4), color=S.INK2, ha="center")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=S.fs(6.4), linespacing=1.1)
    ax.set_xlim(0, 41); ax.set_xlabel("time")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.set_title("V7 burst peak precedes any violation", fontsize=S.fs(8))
    ax.legend(handles=[Line2D([], [], ls="", marker="o", color=S.BLUE, label="burst peak"),
                       Line2D([], [], ls="", marker="X", color=S.ORANGE, label="first physicality violation")],
              loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=1, fontsize=S.fs(6.4))
    S.save(fig, "fig3_size_transfer_envelope")


# ------------------------------------------------------------------ Figure 4: mechanism
def fig4():
    mm = np.load(R + "mechanism_m5.npz", allow_pickle=True)
    cells = list(mm["cells"])
    m4 = mech_m4()
    fig = plt.figure(figsize=(S.DOUBLE, 7.2))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.62)
    # (a) structural pathway -- counts derived from the committed family list
    labs = list(cells[0]["share_all"].keys())
    Q = {"A": 1, "a": -1, "+": 1, "-": -1, "z": 0}
    charged = [l for l in labs if sum(Q[c] for c in l) != 0]
    neutral = [l for l in labs if sum(Q[c] for c in l) == 0]
    assert sorted(neutral) == sorted(["Aaz", "a+z", "+-z", "zzz"]), neutral
    ax = fig.add_subplot(gs[0, 0]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    S.panel(ax, "a", x=-0.12)
    box(ax, -0.1, 0.83, 1.2, 0.15, f"{len(labs)} κ3 families enter the second-order\nequations (M = 5)")
    box(ax, -0.1, 0.59, 1.2, 0.16, f"{len(charged)} carry net excitation: exact κ3 ≡ 0 here\n(excitation conserved)",
        fc="#f0efea")
    box(ax, -0.1, 0.35, 1.2, 0.16, f"{len(neutral)} neutral; +−z and zzz do not couple to the\ntracked equations",
        fc="#f0efea")
    box(ax, -0.1, 0.0, 1.2, 0.27, "only Aaz = ⟨a†a σz⟩c and a+z = ⟨a σ+ σz⟩c\ncan contribute: "
        "a structural property of the\nevaluated hierarchy and symmetry sector", fc="#eaf2fb")
    arrow(ax, 0.5, 0.83, 0.5, 0.75); arrow(ax, 0.5, 0.59, 0.5, 0.51); arrow(ax, 0.5, 0.35, 0.5, 0.27)
    # (b) M = 5 per-draw influence split
    ax = fig.add_subplot(gs[0, 1]); S.panel(ax, "b", x=-0.3)
    for i, reg in enumerate(REGS):
        cs = [c for c in cells if c["regime"] == reg]
        j = jitter(len(cs), 0.13)
        for key, dx, filled in (("share_aaz", -0.2, True), ("share_apz", 0.2, False)):
            v = np.array([c[key] for c in cs])
            ax.scatter(i + dx + j, v, s=S.MS, marker="o", facecolors=S.INK if filled else S.light(S.INK),
                       edgecolors=S.INK, linewidths=0.9, zorder=3)
            ax.plot([i + dx - 0.17, i + dx + 0.17], [np.median(v)] * 2, color=S.BLUE, lw=1.4, zorder=2)
    ax.axhline(0.5, color=S.MUTED, lw=0.6, ls="--"); ax.axhline(0.15, color=S.MUTED, lw=0.6, ls=":")
    ax.text(1.5, 0.51, "equal split", fontsize=S.fs(6.2), color=S.INK2, va="bottom", ha="right")
    ax.text(1.5, 0.16, "C2 floor 0.15", fontsize=S.fs(6.2), color=S.INK2, va="bottom", ha="right")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["low loss", "normal loss"]); ax.set_xlim(-0.5, 1.5); ax.set_ylim(0, 1)
    ax.set_ylabel("share of $D_f$ (whole-dynamics influence)")
    ax.set_title("prospective M = 5: six fresh draws per regime", fontsize=S.fs(8))
    ax.legend(handles=[Line2D([], [], ls="", marker="o", mfc=S.INK, mec=S.INK, label="Aaz  ⟨a†a σz⟩c"),
                       Line2D([], [], ls="", marker="o", mfc=S.light(S.INK), mec=S.INK, label="a+z  ⟨a σ+ σz⟩c"),
                       Line2D([], [], color=S.BLUE, lw=1.4, label="median")], loc="upper center",
              bbox_to_anchor=(0.45, -0.14), fontsize=S.fs(6.4), ncol=1, handletextpad=0.3)
    # (c) exploratory M = 4 vs prospective M = 5 (Aaz share; a+z = 1 - Aaz by structure)
    ax = fig.add_subplot(gs[1, 0]); S.panel(ax, "c", x=-0.2)
    for reg in REGS:
        m5 = np.median([c["share_aaz"] for c in cells if c["regime"] == reg])
        y = [m4[(reg, "Aaz")]["share"], m5]
        filled = S.REGIME[reg]["filled"]
        ax.plot([0, 1], y, color=S.INK2, lw=0.9)
        ax.scatter([0, 1], y, s=44, facecolors=S.INK if filled else S.light(S.INK), edgecolors=S.INK, linewidths=1.1, zorder=3)
        ax.text(1.07, y[1], f"{'low' if reg == 'POCKET' else 'normal'} loss  {y[1]:.2f}", fontsize=S.fs(6.8), va="center")
        ax.text(-0.07, y[0], f"{y[0]:.2f}", fontsize=S.fs(6.8), va="center", ha="right")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["M = 4\nexploratory\n(Result 25)", "M = 5\npre-registered\n(Result 42)"])
    ax.set_xlim(-0.35, 1.65); ax.set_ylim(0.5, 0.8); ax.set_ylabel("Aaz share of $D_f$ (a+z = remainder)")
    ax.set_title("the split reproduces prospectively", fontsize=S.fs(8))
    # (d) closed-loop ablation
    ax = fig.add_subplot(gs[1, 1]); S.panel(ax, "d", x=-0.24)
    TOP = 1.2
    for i, reg in enumerate(REGS):
        cs = [c for c in cells if c["regime"] == reg]
        j = jitter(len(cs), 0.13)
        for key, dx, filled, name in (("loss_aaz", -0.2, True, "Aaz"), ("loss_apz", 0.2, False, "a+z")):
            for jj, c in zip(j, cs):
                v = c[key]
                if np.isinf(v):
                    ax.scatter(i + dx + jj, TOP, marker="x", s=26, c=S.INK, linewidths=1.1, zorder=3)
                else:
                    ax.scatter(i + dx + jj, v, marker="o", s=S.MS, facecolors=S.INK if filled else S.light(S.INK),
                               edgecolors=S.INK, linewidths=0.9, zorder=3)
            ax.text(i + dx, -0.09, name, ha="center", va="top", fontsize=S.fs(6.6), color=S.INK2)
    ax.axhspan(1.1, 1.3, color="#f3f2ee", zorder=0)
    ax.axhline(0, color=S.MUTED, lw=0.6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["low loss", "normal loss"]); ax.tick_params(axis="x", pad=20)
    ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.02, 1.3)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0, TOP]); ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1", "unstable ×"])
    ax.set_ylabel("fraction of V7's advantage lost")
    ax.set_title("both families are load-bearing for frozen V7", fontsize=S.fs(8))
    ax.legend(handles=[Line2D([], [], ls="", marker="o", mfc=S.INK, mec=S.INK, label="Aaz removed"),
                       Line2D([], [], ls="", marker="o", mfc=S.light(S.INK), mec=S.INK, label="a+z removed"),
                       Line2D([], [], ls="", marker="x", color=S.INK, label="ablated rollout unstable")],
              loc="upper center", bbox_to_anchor=(0.45, -0.2), fontsize=S.fs(6.4))
    S.save(fig, "fig4_mechanism")


# ------------------------------------------------------------------ Figure 5: conventional truncations + runtime
def fig5():
    import csv
    t4 = np.load(R + "t4_eval.npz", allow_pickle=True)["res"].item()
    # mean field: categorical, analytic (all coherences vanish for these initial states); committed note
    assert "identically frozen from these ICs" in open(R + "pilot_eval.log").read()
    fig = plt.figure(figsize=(S.DOUBLE, 6.6))
    gs = fig.add_gridspec(2, 2, hspace=0.75, wspace=0.42, height_ratios=[1, 1], width_ratios=[1.1, 1])
    # (a) ladder at M = 3, draws 50-52, both regimes
    ax = fig.add_subplot(gs[0, 0]); S.panel(ax, "a", x=-0.2)
    methods = [("mf", "mean\nfield"), ("plain", "second\norder"), ("t3", "third\n(κ4 = 0)"),
               ("t4", "fourth\n(κ5 = 0)"), ("v7", "learned\n(V7)")]
    TOP = 45
    for i, (key, _) in enumerate(methods):
        if key == "mf":
            ax.text(i, 0, "frozen:\nno burst", ha="center", va="center", fontsize=S.fs(6.4), color=S.INK2,
                    bbox=dict(fc="white", ec="none", pad=1.0), zorder=5)
            continue
        for reg in REGS:
            for k, sd in enumerate((50, 51, 52)):
                val, ok = t4[(3, reg, sd)][key][:2]
                x = i + (-0.14 if reg == "POCKET" else 0.14) + (k - 1) * 0.06
                if not ok or not np.isfinite(val):
                    ax.scatter(x, TOP, marker="x", s=26, c=S.METHOD[key]["color"], linewidths=1.2, zorder=3)
                else:
                    ax.scatter(x, 100 * val, s=S.MS, **S.mk(key, reg))
    ax.axhspan(38, 52, color="#f3f2ee", zorder=0)
    ax.axhline(0, color=S.MUTED, lw=0.6)
    ax.set_xticks(range(len(methods))); ax.set_xticklabels([m[1] for m in methods], fontsize=S.fs(6.8))
    ax.set_xlim(-0.6, 4.6); ax.set_ylim(-40, 52); ax.set_yticks([-40, -20, 0, 20, TOP]); ax.set_yticklabels(["−40", "−20", "0", "20", "runaway ×"])
    ax.set_ylabel("signed burst-peak error (%)")
    ax.set_title("truncation ladder, M = 3 (3 draws × 2 regimes)", fontsize=S.fs(8))
    # (b) runtime (M = 3 benchmark): build, rollout, total medians over cells
    rt = list(csv.DictReader(open(R + "runtime_baseline_comparison.csv")))
    ax = fig.add_subplot(gs[0, 1]); S.panel(ax, "b", x=-0.3)
    order = [("2nd-order", "plain", "second order"), ("V7", "v7", "V7"), ("kappa4=0 (T3)", "t3", "κ4 = 0"),
             ("kappa5=0 (T4)", "t4", "κ5 = 0")]
    comps = [("build_med", "|", -0.22), ("roll_med", "o", 0.0), ("total_med", "s", 0.22)]
    for i, (name, key, lab) in enumerate(order):
        rr = [r for r in rt if r["method"] == name]
        assert len(rr) == 6, (name, len(rr))
        col = S.METHOD[key]["color"]
        for comp, mark, dy in comps:
            v = np.median([float(r[comp]) for r in rr])
            if mark == "|":
                ax.scatter(v, i + dy, marker="|", s=50, c=col, linewidths=1.3, zorder=3)
            else:
                ax.scatter(v, i + dy, marker=mark, s=S.MS, facecolors=S.light(col) if mark == "o" else col,
                           edgecolors=col, linewidths=1.2, zorder=3)
            if comp == "total_med":
                ax.text(v * 1.35, i + dy, f"{v:.1f} s", fontsize=S.fs(6.3), va="center", color=S.INK2)
    ax.set_xscale("log"); ax.set_xlim(0.008, 400)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([o[2] for o in order])
    ax.invert_yaxis(); ax.set_xlabel("wall-clock seconds (median, M = 3, one machine)")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.set_title("cost: build, rollout, total", fontsize=S.fs(8))
    ax.legend(handles=[Line2D([], [], ls="", marker="|", ms=7, mew=1.3, color=S.INK2, label="build"),
                       Line2D([], [], ls="", marker="o", mfc=S.light(S.INK2), mec=S.INK2, label="rollout"),
                       Line2D([], [], ls="", marker="s", color=S.INK2, label="total")],
              loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3, fontsize=S.fs(6.4))
    # (c) representative traces with all truncations
    d = np.load(R + "baseline_burst_trace_M3_pocket_draw50.npz", allow_pickle=True)
    meta = d["meta"].item()
    assert meta["t4_guard_triggered"] and not meta["t3_guard_triggered"]
    ax = fig.add_subplot(gs[1, :]); S.panel(ax, "c", x=-0.08)
    for key, tk, nk in (("exact", "ts_exact", "n_exact"), ("plain", "ts_plain", "n_plain"), ("t3", "ts_t3", "n_t3"),
                        ("t4", "ts_t4", "n_t4"), ("v7", "ts_v7", "n_v7")):
        m = S.METHOD[key]
        ax.plot(d[tk], d[nk], color=m["color"], ls=m["ls"], lw=2.2 if key == "exact" else 1.4, label=m["label"],
                zorder=3 if key == "v7" else 2)
    ax.axvline(meta["t4_stop"], color=S.YELLOW, lw=0.7, ls=":")
    ax.text(meta["t4_stop"] + 0.2, 4.3, "fourth order turns unphysical (n < 0) and\ndiverges; divergence guard stops it at t = %.1f"
            % meta["t4_stop"], fontsize=S.fs(6.4), color=S.INK2, va="top")
    ax.set_xlim(0, 20); ax.set_ylim(0, 4.7); ax.set_xlabel("time"); ax.set_ylabel("⟨a†a⟩")
    ax.set_title("one evaluated cell (M = 3, low loss, draw 50)", fontsize=S.fs(8))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=S.fs(6.6))
    S.save(fig, "fig5_truncations_runtime")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5()
    print("built figures/fig1..fig5 (.pdf, .eps, .png)")
