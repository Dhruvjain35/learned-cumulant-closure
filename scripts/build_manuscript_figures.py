"""
Build the five main manuscript figures from COMMITTED artifacts only. Render-only: no exact solve, no closure
rollout, no network evaluation. Run from the repo root:  python run.py scripts/build_manuscript_figures.py
(or `make figures`). Writes figures/fig{1..5}_*.pdf and .png. Panel -> artifact map: docs/FIGURE_PROVENANCE.md.

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
def box(ax, x, y, w, h, text, fc="#f4f3ef", ec=S.INK2, fs=7.6, bold=False, color=S.INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.02", fc=fc, ec=ec, lw=0.7))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x0, y0, x1, y1, color=S.INK2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, lw=0.8, color=color))


def jitter(n, w=0.18, seed=0):
    return np.linspace(-w, w, n) if n > 1 else np.zeros(1)


# ------------------------------------------------------------------ Figure 1: problem and method
def fig1():
    from training_support import TRAIN
    fig = plt.figure(figsize=(S.DOUBLE, 5.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1.05], hspace=0.32, wspace=0.22)
    # (a) model schematic
    ax = fig.add_subplot(gs[0, 0]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); S.panel(ax, "a", x=-0.02)
    for xm, th in ((0.17, (300, 60)), (0.83, (120, 240))):
        ax.add_patch(Arc((xm, 0.6), 0.16, 0.5, theta1=th[0], theta2=th[1], lw=2.2, color=S.INK2))
    xs = np.linspace(0.3, 0.7, 5)
    ys = 0.6 + 0.06 * np.array([1, -1, 0.5, -0.6, 0.9])
    for x, y in zip(xs, ys):
        ax.add_patch(Circle((x, y), 0.026, fc=S.BLUE, ec=S.INK, lw=0.6))
    ax.text(0.5, 0.97, "one cavity mode $a$ + $M$ two-level emitters\nfrequencies $\\omega_i$ with Gaussian disorder σ",
            ha="center", va="top", fontsize=7.2)
    arrow(ax, 0.88, 0.6, 0.97, 0.6); ax.text(0.925, 0.64, "κ", ha="center", fontsize=9)
    ax.annotate("", xy=(xs[1], ys[1] - 0.12), xytext=(xs[1], ys[1] - 0.035),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color=S.INK2))
    ax.text(xs[1] + 0.025, ys[1] - 0.13, "γ↓", fontsize=8.5)
    ax.text(xs[3] - 0.01, ys[3] - 0.13, "γφ", fontsize=8.5)
    ax.text(0.5, 0.30, "$g_i = g_c/\\sqrt{M}$ (collective coupling fixed)\n"
            "start: cavity vacuum, all emitters excited\n"
            "observable: burst peak $F_{\\rm peak} = \\kappa\\,\\max_t\\langle a^\\dagger a\\rangle$",
            ha="center", va="top", fontsize=7.2, linespacing=1.5)
    # (b) closure problem
    ax = fig.add_subplot(gs[0, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); S.panel(ax, "b", x=-0.02)
    box(ax, 0.0, 0.76, 0.45, 0.16, "tracked moments\n(orders 1 and 2)", fs=7)
    box(ax, 0.55, 0.76, 0.45, 0.16, "their exact equations\nneed order-3 moments", fs=7)
    arrow(ax, 0.45, 0.84, 0.55, 0.84)
    box(ax, 0.0, 0.40, 0.45, 0.24, "second order:\nset connected κ3 = 0\n(≈ +29% burst overshoot)", fs=7, fc="#fbeee8")
    box(ax, 0.55, 0.40, 0.45, 0.24, "learned closure:\npredict the missing\nconnected κ3 in the ODE", fs=7, fc="#eaf2fb")
    arrow(ax, 0.70, 0.76, 0.30, 0.64); arrow(ax, 0.775, 0.76, 0.775, 0.64)
    box(ax, 0.06, 0.02, 0.88, 0.24, "order-3 moment = 2nd-order factorization + κ3\n"
        "same engine; only κ3 is supplied", fs=7)
    arrow(ax, 0.225, 0.40, 0.35, 0.26); arrow(ax, 0.775, 0.40, 0.65, 0.26)
    # (c) architecture
    ax = fig.add_subplot(gs[1, 0]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); S.panel(ax, "c", x=-0.02)
    feats = ["lower moments of the\nmonomial's own factors", "site detunings Δ/κ",
             "intensive context: n/M,\n⟨σz⟩, coherence magnitude", "regime: $g_c/\\kappa$, κ, γφ/κ",
             "memory traces of context\n(memory-aware branch only)"]
    for i, t in enumerate(feats):
        y0 = 0.82 - i * 0.19
        box(ax, 0.0, y0, 0.47, 0.15, t, fs=6.6, fc="#f7f6f2")
        arrow(ax, 0.47, y0 + 0.075, 0.55, 0.5)
    box(ax, 0.55, 0.36, 0.24, 0.28, "one shared\nMLP per\ncumulant\ntype", fs=6.6, fc="#eaf2fb")
    arrow(ax, 0.79, 0.5, 0.84, 0.5)
    box(ax, 0.84, 0.40, 0.16, 0.20, "κ3", fs=8.5)
    ax.text(0.775, 0.30, "2×128 ReLU; output clipped to\ntraining range; same network\nfor every site and every M",
            ha="center", va="top", fontsize=6.3, color=S.INK2)
    # (d) router + training support in the (kappa, gamma_phi) plane
    ax = fig.add_subplot(gs[1, 1]); S.panel(ax, "d", x=-0.16)
    router = json.load(open("portable_v7/router.json"))
    kmax, gmax = router["kappa_max"], router["gamma_phi_max"]
    ax.add_patch(Rectangle((0.018, 0.0), kmax - 0.018, gmax, fc="#eaf2fb", ec=S.BLUE, lw=0.9, zorder=0))
    ax.text(kmax * 1.04, 0.004, "← memory-aware branch", fontsize=6.4, color=S.BLUE, va="bottom")
    ax.text(0.47, 0.128, "memoryless branch elsewhere", fontsize=6.4, color=S.INK2, ha="right", va="top")
    kg = sorted({(t[1], t[2]) for t in TRAIN})
    ax.scatter([k for k, g in kg], [g for k, g in kg], marker="o", s=14, c=S.INK2, zorder=3)
    cells = json.load(open("results/heldout_param_cells.json"))["cells"]
    ax.scatter([c["kappa"] for c in cells], [c["gamma_phi"] for c in cells], marker="o", s=18, facecolors="none",
               edgecolors=S.BLUE, linewidths=0.9, zorder=4)
    ax.scatter([0.03, 0.15], [0.02, 0.02], marker="*", s=80, c=S.INK, zorder=5)
    ax.set_xscale("log"); ax.set_xlim(0.018, 0.5); ax.set_ylim(0, 0.13)
    ax.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5]); ax.set_xticklabels(["0.02", "0.05", "0.1", "0.2", "0.5"])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("cavity loss κ"); ax.set_ylabel("pure dephasing γφ")
    ax.set_title("fixed router: decided once at t = 0 from (κ, γφ)", fontsize=7.8)
    ax.legend(handles=[Line2D([], [], ls="", marker="o", mfc=S.INK2, mec=S.INK2, ms=4, label="training (κ, γφ)"),
                       Line2D([], [], ls="", marker="o", mfc="none", mec=S.BLUE, ms=4.5, label="held-out cells (Result 40)"),
                       Line2D([], [], ls="", marker="*", color=S.INK, ms=8, label="low / normal loss headline cells")],
              loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6.3, handletextpad=0.2, columnspacing=0.8)
    S.save(fig, "fig1_problem_method")


# ------------------------------------------------------------------ Figure 2: accuracy and parameter transfer
def fig2():
    fig = plt.figure(figsize=(S.DOUBLE, 5.1))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.28)
    # (a) representative trace (canonical cell: M=3 low loss draw 50, Results 24/27)
    d = np.load(R + "baseline_burst_trace_M3_pocket_draw50.npz", allow_pickle=True)
    ax = fig.add_subplot(gs[0, 0]); S.panel(ax, "a")
    for key, tk, nk in (("exact", "ts_exact", "n_exact"), ("plain", "ts_plain", "n_plain"), ("v7", "ts_v7", "n_v7")):
        m = S.METHOD[key]
        ax.plot(d[tk], d[nk], color=m["color"], ls=m["ls"], lw=1.6 if key != "exact" else 2.2,
                label=m["label"], zorder=3 if key == "v7" else 2)
    ax.set_xlim(0, 15); ax.set_xlabel("time t"); ax.set_ylabel("cavity photon number ⟨a†a⟩")
    ax.set_title("one evaluated cell: M = 3, low loss, draw 50", fontsize=8)
    ax.legend(loc="upper right")
    # (b) original exact M<=5 benchmark
    bp = burst_pred()
    ax = fig.add_subplot(gs[0, 1]); S.panel(ax, "b")
    for meth in ("plain", "v7"):
        for reg in REGS:
            off = -0.09 if reg == "POCKET" else 0.09
            for M in (2, 3, 4, 5):
                vals = [abs(bp[(reg, M, s)][meth]) for s in (50, 51, 52)]
                ax.scatter(np.full(3, M + off), vals, s=16, **S.mk(meth, reg))
    ax.set_yscale("log"); ax.set_ylim(5e-5, 1); ax.set_xticks([2, 3, 4, 5])
    ax.set_xlabel("emitters M (in training)"); ax.set_ylabel("|burst-peak error|")
    ax.set_title("original exact cells: held-out disorder draws only", fontsize=8)
    ax.text(5.35, 0.29, "second order", color=S.ORANGE, fontsize=7, va="center")
    ax.text(5.35, 0.004, "V7", color=S.BLUE, fontsize=7, va="center")
    ax.set_xlim(1.6, 6.0)
    ax.legend(handles=[Line2D([], [], ls="", marker="o", mfc=S.INK2, mec=S.INK2, label="low loss (filled)"),
                       Line2D([], [], ls="", marker="o", mfc="white", mec=S.INK2, label="normal loss (hollow)")],
              loc="lower left", fontsize=6.6)
    # (c) Result 40 -- all 30 cells
    hp = np.load(R + "heldout_param_burst.npz", allow_pickle=True)
    rows, stats = hp["rows"], hp["stats"].item()
    ax = fig.add_subplot(gs[1, 0]); S.panel(ax, "c")
    classes = [("B", "interpolation"), ("C", "one-axis\nextrapolation"), ("D", "two-axis\nextrapolation")]
    for i, (cls, _) in enumerate(classes):
        rs = [r for r in rows if r["cls"] == cls]
        j = jitter(len(rs), 0.13)
        for meth, dx in (("plain", -0.18), ("v7", 0.18)):
            vals = np.array([r[meth]["abs"] for r in rs])
            ax.scatter(i + dx + j * 0.6, vals, s=13, **S.mk(meth, None, alpha=0.9))
            med, q1, q3 = np.median(vals), np.percentile(vals, 25), np.percentile(vals, 75)
            ax.plot([i + dx - 0.13, i + dx + 0.13], [med, med], color=S.INK, lw=1.4, zorder=4)
            ax.plot([i + dx, i + dx], [q1, q3], color=S.INK, lw=0.8, zorder=4)
    ax.set_yscale("log"); ax.set_ylim(5e-5, 1); ax.set_xticks(range(3)); ax.set_xticklabels([c[1] for c in classes])
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylabel("|burst-peak error|")
    ax.set_title("30 parameter points absent from training, M = 5", fontsize=8)
    ax.legend(handles=[Line2D([], [], ls="", marker="s", mfc=S.ORANGE, mec=S.ORANGE, label="second order"),
                       Line2D([], [], ls="", marker="o", mfc=S.BLUE, mec=S.BLUE, label="learned closure (V7)"),
                       Line2D([], [], color=S.INK, lw=1.4, label="median (bar: IQR)")],
              loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3, fontsize=6.4, columnspacing=0.8)
    # (d) summary table (from committed stats)
    ax = fig.add_subplot(gs[1, 1]); ax.axis("off"); S.panel(ax, "d", x=-0.02)
    hdr = ["support class", "N", "V7 median", "V7 IQR", "2nd-order median"]
    body = []
    for cls, name in (("B", "interpolation"), ("C", "one-axis extrap."), ("D", "two-axis extrap.")):
        s = stats[cls]
        body.append([name, str(s["n"]), f"{100 * s['v7_abs']['median']:.2f}%",
                     f"{100 * s['v7_abs']['q25']:.2f}–{100 * s['v7_abs']['q75']:.2f}%",
                     f"{100 * s['plain_abs']['median']:.1f}%"])
    a = stats["all"]
    worst = max(r["v7"]["abs"] for r in rows)
    body.append(["all 30", str(a["n"]), f"{100 * a['v7_abs']['median']:.2f}%", "", f"{100 * a['plain_abs']['median']:.1f}%"])
    tb = ax.table(cellText=body, colLabels=hdr, cellLoc="center", colLoc="center",
                  colWidths=[0.27, 0.07, 0.18, 0.23, 0.25], bbox=[0.0, 0.34, 1.0, 0.62])
    tb.auto_set_font_size(False); tb.set_fontsize(7)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor("#d8d7d2"); cell.set_linewidth(0.5)
        if r == 0:
            cell.set_text_props(fontweight="bold")
    wins = sum(r["win"] for r in rows)
    ax.text(0.5, 0.0, f"V7 closer to exact in {wins}/30 cells; worst V7 error {100 * worst:.2f}%;\n"
            "V7 physical in all 30 cells. Classes are defined in raw physical parameters;\n"
            "in the model's ratio-to-κ inputs only 7 of 30 lie outside the trained ranges.",
            ha="center", va="bottom", fontsize=6.9, transform=ax.transAxes)
    S.save(fig, "fig2_accuracy_parameter_transfer")


# ------------------------------------------------------------------ Figure 3: size transfer and validity envelope
def fig3():
    lad, dm = piqs_ladder(), disordered_m12()
    fig = plt.figure(figsize=(S.DOUBLE, 5.2))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.62, width_ratios=[1.1, 1])
    # (a) signed error vs M (zero-disorder exact permutation-invariant ladder), measured points only
    ax = fig.add_subplot(gs[0, 0]); S.panel(ax, "a", x=-0.2)
    Ms = [5, 8, 10, 12, 16]
    for reg in REGS:
        for meth in ("plain", "v7"):
            y = [100 * lad[(reg, M)][meth] for M in Ms]
            m = S.METHOD[meth]
            ax.plot(Ms, y, color=m["color"], ls=m["ls"], lw=1.0, alpha=0.8, zorder=2)
            ax.scatter(Ms, y, s=20, **S.mk(meth, reg))
    for reg in REGS:   # disordered M = 12, one draw, quantum trajectories (MC error bar)
        for meth in ("plain", "v7"):
            v = 100 * dm[reg][meth]
            ax.errorbar(12.55 if reg == "POCKET" else 12.85, v, yerr=100 * dm[reg]["mc"], fmt="none",
                        ecolor=S.METHOD[meth]["color"], elinewidth=0.9, capsize=2, zorder=2)
            ax.scatter(12.55 if reg == "POCKET" else 12.85, v, s=26, marker="x", c=S.METHOD[meth]["color"], zorder=4)
    ax.axhline(0, color=S.MUTED, lw=0.6)
    ax.set_xticks(Ms); ax.set_xlabel("emitters M (trained at M ≤ 5)"); ax.set_ylabel("signed burst-peak error (%)")
    ax.set_title("size transfer: measured points only", fontsize=8)
    ax.text(16.3, 100 * lad[("CONTROL", 16)]["plain"], "second order", color=S.ORANGE, fontsize=7, va="center")
    ax.text(16.3, 100 * lad[("POCKET", 16)]["v7"], "V7", color=S.BLUE, fontsize=7, va="center")
    ax.set_xlim(4, 18.2)
    ax.legend(handles=[Line2D([], [], ls="", marker="o", mfc=S.INK2, mec=S.INK2, label="σ = 0, exact (PIQS); low loss filled"),
                       Line2D([], [], ls="", marker="o", mfc="white", mec=S.INK2, label="σ = 0, exact; normal loss hollow"),
                       Line2D([], [], ls="", marker="x", color=S.INK2, label="σ = 0.15, trajectories, one draw (±MC)")],
              loc="center left", fontsize=6.2, bbox_to_anchor=(0.0, 0.58))
    # (b) M = 12: disordered trajectories vs zero-disorder exact
    ax = fig.add_subplot(gs[0, 1]); S.panel(ax, "b", x=-0.2)
    for i, reg in enumerate(REGS):
        for meth, dx in (("plain", -0.12), ("v7", 0.12)):
            a = 100 * lad[(reg, 12)][meth]; b = 100 * dm[reg][meth]
            ax.scatter(i + dx - 0.05, a, s=26, **S.mk(meth, None))
            ax.errorbar(i + dx + 0.05, b, yerr=100 * dm[reg]["mc"], fmt="none", ecolor=S.METHOD[meth]["color"], capsize=2)
            ax.scatter(i + dx + 0.05, b, s=30, marker="x", c=S.METHOD[meth]["color"], zorder=4)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["low loss", "normal loss"]); ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("signed error at M = 12 (%)"); ax.set_ylim(0, 34)
    ax.set_title("M = 12: σ = 0 exact vs σ = 0.15 trajectories", fontsize=8)
    ax.legend(handles=[Line2D([], [], ls="", marker="s", mfc=S.INK2, mec=S.INK2, label="σ = 0, exact permutation-invariant"),
                       Line2D([], [], ls="", marker="x", color=S.INK2, label="σ = 0.15, 400 trajectories, one draw")],
              loc="center right", fontsize=6.2)
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
                ax.scatter(j, y + 0.12, marker=sym, s=60, facecolors=col if (filled or sym == "X") else "white",
                           edgecolors=col, linewidths=1.2, zorder=3)
            ax.text(j, y - 0.27, txt, ha="center", va="center", fontsize=6.6, color=S.INK2)
    ax.set_yticks(range(len(rowsdef))); ax.set_yticklabels([r[0] for r in rowsdef][::-1], fontsize=7)
    ax.set_xticks(range(len(Mcols))); ax.set_xticklabels([str(M) for M in Mcols]); ax.set_xlabel("emitters M")
    ax.set_xlim(-0.5, len(Mcols) - 0.5); ax.set_ylim(-0.6, len(rowsdef) - 0.4)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.set_title("V7 physicality over t ≤ 40 (physical / tested)", fontsize=8)
    ax.legend(handles=[Line2D([], [], ls="", marker="o", mfc=S.BLUE, mec=S.BLUE, label="all tested draws physical"),
                       Line2D([], [], ls="", marker="o", mfc="white", mec=S.YELLOW, mew=1.2, label="draw-dependent"),
                       Line2D([], [], ls="", marker="X", mfc=S.ORANGE, mec=S.ORANGE, label="leaves physical set"),
                       Line2D([], [], ls="", marker="", label="– not measured")],
              loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=4, fontsize=6.2, handletextpad=0.2, columnspacing=0.8)
    # (d) burst time vs first violation time
    ax = fig.add_subplot(gs[1, 1]); S.panel(ax, "d", x=-0.4)
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
            ax.text(22, y + 0.18, "no violation to t = 40", fontsize=6.4, color=S.INK2, ha="center")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=6.4, linespacing=1.1)
    ax.set_xlim(0, 41); ax.set_xlabel("time")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.set_title("V7 burst peak precedes any violation", fontsize=8)
    ax.legend(handles=[Line2D([], [], ls="", marker="o", color=S.BLUE, label="burst peak"),
                       Line2D([], [], ls="", marker="X", color=S.ORANGE, label="first physicality violation")],
              loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=6.4)
    S.save(fig, "fig3_size_transfer_envelope")


# ------------------------------------------------------------------ Figure 4: mechanism
def fig4():
    mm = np.load(R + "mechanism_m5.npz", allow_pickle=True)
    cells = list(mm["cells"])
    m4 = mech_m4()
    fig = plt.figure(figsize=(S.DOUBLE, 5.3))
    gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.38)
    # (a) structural pathway -- counts derived from the committed family list
    labs = list(cells[0]["share_all"].keys())
    Q = {"A": 1, "a": -1, "+": 1, "-": -1, "z": 0}
    charged = [l for l in labs if sum(Q[c] for c in l) != 0]
    neutral = [l for l in labs if sum(Q[c] for c in l) == 0]
    assert sorted(neutral) == sorted(["Aaz", "a+z", "+-z", "zzz"]), neutral
    ax = fig.add_subplot(gs[0, 0]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); S.panel(ax, "a", x=-0.02)
    box(ax, 0.12, 0.78, 0.76, 0.17, f"{len(labs)} κ3 families enter the\nsecond-order equations (M = 5)", fs=6.8)
    box(ax, 0.0, 0.44, 0.48, 0.22, f"{len(charged)} carry net excitation:\nexact κ3 ≡ 0 here\n(excitation conserved)",
        fs=6.6, fc="#f0efea")
    box(ax, 0.52, 0.44, 0.48, 0.22, f"{len(neutral)} neutral; +−z and zzz\ndo not couple to the\ntracked equations",
        fs=6.6, fc="#f0efea")
    box(ax, 0.06, 0.0, 0.88, 0.27, "only Aaz = ⟨a†a σz⟩c and a+z = ⟨a σ+ σz⟩c\n"
        "can contribute: a structural property of\nthe evaluated hierarchy and symmetry sector", fs=6.6, fc="#eaf2fb")
    arrow(ax, 0.35, 0.78, 0.24, 0.66); arrow(ax, 0.65, 0.78, 0.76, 0.66); arrow(ax, 0.76, 0.44, 0.6, 0.27)
    # (b) M = 5 per-draw influence split
    ax = fig.add_subplot(gs[0, 1]); S.panel(ax, "b", x=-0.3)
    for i, reg in enumerate(REGS):
        cs = [c for c in cells if c["regime"] == reg]
        j = jitter(len(cs), 0.13)
        for key, dx, filled in (("share_aaz", -0.2, True), ("share_apz", 0.2, False)):
            v = np.array([c[key] for c in cs])
            ax.scatter(i + dx + j, v, s=20, marker="o", facecolors=S.INK if filled else "white",
                       edgecolors=S.INK, linewidths=0.9, zorder=3)
            ax.plot([i + dx - 0.17, i + dx + 0.17], [np.median(v)] * 2, color=S.BLUE, lw=1.4, zorder=2)
    ax.axhline(0.5, color=S.MUTED, lw=0.6, ls="--"); ax.axhline(0.15, color=S.MUTED, lw=0.6, ls=":")
    ax.text(1.5, 0.51, "equal split", fontsize=6.2, color=S.INK2, va="bottom", ha="right")
    ax.text(1.5, 0.16, "C2 floor 0.15", fontsize=6.2, color=S.INK2, va="bottom", ha="right")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["low loss", "normal loss"]); ax.set_xlim(-0.5, 1.5); ax.set_ylim(0, 1)
    ax.set_ylabel("share of $D_f$ (whole-dynamics influence)")
    ax.set_title("prospective M = 5: six fresh draws per regime", fontsize=8)
    ax.legend(handles=[Line2D([], [], ls="", marker="o", mfc=S.INK, mec=S.INK, label="Aaz  ⟨a†a σz⟩c"),
                       Line2D([], [], ls="", marker="o", mfc="white", mec=S.INK, label="a+z  ⟨a σ+ σz⟩c"),
                       Line2D([], [], color=S.BLUE, lw=1.4, label="median")], loc="upper right", fontsize=6.4, ncol=3,
              columnspacing=0.8, handletextpad=0.3)
    # (c) exploratory M = 4 vs prospective M = 5 (Aaz share; a+z = 1 - Aaz by structure)
    ax = fig.add_subplot(gs[1, 0]); S.panel(ax, "c", x=-0.2)
    for reg in REGS:
        m5 = np.median([c["share_aaz"] for c in cells if c["regime"] == reg])
        y = [m4[(reg, "Aaz")]["share"], m5]
        filled = S.REGIME[reg]["filled"]
        ax.plot([0, 1], y, color=S.INK2, lw=0.9)
        ax.scatter([0, 1], y, s=34, facecolors=S.INK if filled else "white", edgecolors=S.INK, linewidths=1, zorder=3)
        ax.text(1.07, y[1], f"{'low' if reg == 'POCKET' else 'normal'} loss  {y[1]:.2f}", fontsize=6.8, va="center")
        ax.text(-0.07, y[0], f"{y[0]:.2f}", fontsize=6.8, va="center", ha="right")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["M = 4\nexploratory (Result 25)", "M = 5\npre-registered (Result 42)"])
    ax.set_xlim(-0.35, 1.65); ax.set_ylim(0.5, 0.8); ax.set_ylabel("Aaz share of $D_f$ (a+z = remainder)")
    ax.set_title("the split reproduces prospectively", fontsize=8)
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
                    ax.scatter(i + dx + jj, v, marker="o", s=20, facecolors=S.INK if filled else "white",
                               edgecolors=S.INK, linewidths=0.9, zorder=3)
            ax.text(i + dx, -0.09, name, ha="center", va="top", fontsize=6.6, color=S.INK2)
    ax.axhspan(1.1, 1.3, color="#f3f2ee", zorder=0)
    ax.axhline(0, color=S.MUTED, lw=0.6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["low loss", "normal loss"]); ax.tick_params(axis="x", pad=11)
    ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.02, 1.3)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0, TOP]); ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1", "unstable ×"])
    ax.set_ylabel("fraction of V7's advantage lost")
    ax.set_title("both families are load-bearing for frozen V7", fontsize=8)
    ax.legend(handles=[Line2D([], [], ls="", marker="o", mfc=S.INK, mec=S.INK, label="Aaz removed"),
                       Line2D([], [], ls="", marker="o", mfc="white", mec=S.INK, label="a+z removed"),
                       Line2D([], [], ls="", marker="x", color=S.INK, label="ablated rollout unstable")],
              loc="center left", bbox_to_anchor=(0.0, 0.6), fontsize=6.4)
    S.save(fig, "fig4_mechanism")


# ------------------------------------------------------------------ Figure 5: conventional truncations + runtime
def fig5():
    import csv
    t4 = np.load(R + "t4_eval.npz", allow_pickle=True)["res"].item()
    # mean field: categorical, analytic (all coherences vanish for these initial states); committed note
    assert "identically frozen from these ICs" in open(R + "pilot_eval.log").read()
    fig = plt.figure(figsize=(S.DOUBLE, 5.4))
    gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.42, height_ratios=[1, 1], width_ratios=[1.1, 1])
    # (a) ladder at M = 3, draws 50-52, both regimes
    ax = fig.add_subplot(gs[0, 0]); S.panel(ax, "a", x=-0.2)
    methods = [("mf", "mean\nfield"), ("plain", "second\norder"), ("t3", "third\n(κ4 = 0)"),
               ("t4", "fourth\n(κ5 = 0)"), ("v7", "learned\n(V7)")]
    TOP = 45
    for i, (key, _) in enumerate(methods):
        if key == "mf":
            ax.text(i, 0, "frozen:\nno burst", ha="center", va="center", fontsize=6.4, color=S.INK2)
            continue
        for reg in REGS:
            for k, sd in enumerate((50, 51, 52)):
                val, ok = t4[(3, reg, sd)][key][:2]
                x = i + (-0.14 if reg == "POCKET" else 0.14) + (k - 1) * 0.06
                if not ok or not np.isfinite(val):
                    ax.scatter(x, TOP, marker="x", s=26, c=S.METHOD[key]["color"], linewidths=1.2, zorder=3)
                else:
                    ax.scatter(x, 100 * val, s=18, **S.mk(key, reg))
    ax.axhspan(38, 52, color="#f3f2ee", zorder=0)
    ax.axhline(0, color=S.MUTED, lw=0.6)
    ax.set_xticks(range(len(methods))); ax.set_xticklabels([m[1] for m in methods], fontsize=6.8)
    ax.set_xlim(-0.6, 4.6); ax.set_ylim(-40, 52); ax.set_yticks([-40, -20, 0, 20, TOP]); ax.set_yticklabels(["−40", "−20", "0", "20", "runaway ×"])
    ax.set_ylabel("signed burst-peak error (%)")
    ax.set_title("truncation ladder, M = 3 (3 draws × 2 regimes)", fontsize=8)
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
                ax.scatter(v, i + dy, marker=mark, s=22, facecolors="white" if mark == "o" else col,
                           edgecolors=col, linewidths=1.2, zorder=3)
            if comp == "total_med":
                ax.text(v * 1.35, i + dy, f"{v:.1f} s", fontsize=6.3, va="center", color=S.INK2)
    ax.set_xscale("log"); ax.set_xlim(0.008, 400)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([o[2] for o in order])
    ax.invert_yaxis(); ax.set_xlabel("wall-clock seconds (median, M = 3, one machine)")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.set_title("cost: build, rollout, total", fontsize=8)
    ax.legend(handles=[Line2D([], [], ls="", marker="|", ms=7, mew=1.3, color=S.INK2, label="build"),
                       Line2D([], [], ls="", marker="o", mfc="white", mec=S.INK2, label="rollout"),
                       Line2D([], [], ls="", marker="s", color=S.INK2, label="total")],
              loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3, fontsize=6.4)
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
            % meta["t4_stop"], fontsize=6.4, color=S.INK2, va="top")
    ax.set_xlim(0, 20); ax.set_ylim(0, 4.7); ax.set_xlabel("time"); ax.set_ylabel("⟨a†a⟩")
    ax.set_title("one evaluated cell (M = 3, low loss, draw 50)", fontsize=8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=5, fontsize=6.6)
    S.save(fig, "fig5_truncations_runtime")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5()
    print("built figures/fig1..fig5 (.pdf, .png)")
