"""
PLOT ONLY. Reads results/runtime_baseline_comparison.csv (produced by benchmark_rollout_runtime.py) and renders
a build-vs-rollout wall-clock comparison per method (log scale). Does not run or alter any model.

Run:  python run.py scripts/plot_runtime_benchmark.py   (needs numpy + matplotlib)
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["2nd-order", "V7", "kappa4=0 (T3)", "kappa5=0 (T4)"]
LABEL = {"2nd-order": "2nd-order", "V7": "V7 (learned)", "kappa4=0 (T3)": r"$\kappa_4{=}0$ (T3)",
         "kappa5=0 (T4)": r"$\kappa_5{=}0$ (T4)"}

rows = list(csv.DictReader(open("results/runtime_baseline_comparison.csv")))


def agg(method, key):
    vals = [float(r[key]) for r in rows if r["method"] == method and r.get(key) not in ("", None)]
    return vals


build_med = {m: np.median(agg(m, "build_med")) for m in ORDER}
roll_med = {m: np.median(agg(m, "roll_med")) for m in ORDER}
total_med = {m: np.median(agg(m, "total_med")) for m in ORDER}
build_lo = {m: min(agg(m, "build_min")) for m in ORDER}
build_hi = {m: max(agg(m, "build_max")) for m in ORDER}

x = np.arange(len(ORDER)); wdt = 0.38
fig, ax = plt.subplots(figsize=(8.4, 5.0))
bb = ax.bar(x - wdt / 2, [build_med[m] for m in ORDER], wdt, color="#264653", label="build (one-time)")
br = ax.bar(x + wdt / 2, [roll_med[m] for m in ORDER], wdt, color="#2a9d8f", label="rollout")
ax.set_yscale("log")
ax.set_ylim(0.012, 600)                              # headroom so labels clear the title
ax.set_ylabel("wall-clock time  (s, log scale)")
ax.set_xticks(x); ax.set_xticklabels([LABEL[m] for m in ORDER])
ax.set_title("Frozen-runtime benchmark — M=3 cells (median over the evaluated cells)", fontsize=10.5, pad=12)
# total labels just above each build bar
for xi, m in zip(x, ORDER):
    ax.annotate(f"≈{total_med[m]:.1f} s total", xy=(xi, build_med[m]), xytext=(xi, build_med[m] * 1.35),
                ha="center", va="bottom", fontsize=8.5, color="#333")
# T4 build across-cell range caveat
ax.annotate(f"T4 build {build_lo['kappa5=0 (T4)']:.0f}–{build_hi['kappa5=0 (T4)']:.0f} s across cells\n"
            f"(up to ~1068 s observed elsewhere)",
            xy=(3 - wdt / 2, build_med["kappa5=0 (T4)"]), xytext=(1.35, 240),
            ha="center", fontsize=8, color="#9b2226",
            arrowprops=dict(arrowstyle="->", color="#9b2226", lw=0.8))
ax.grid(axis="y", alpha=0.25, which="both")
ax.legend(fontsize=9, loc="upper left")
fig.tight_layout()
fig.savefig("results/runtime_benchmark_fig.png", dpi=150)
print("wrote results/runtime_benchmark_fig.png")
print("medians (s):")
for m in ORDER:
    print(f"  {m:16s} build={build_med[m]:.2f}  rollout={roll_med[m]:.3f}  total={total_med[m]:.2f}")
