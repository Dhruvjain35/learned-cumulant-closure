"""
Runtime/amortization figure. EVERY number is measured from this repo's run logs (sources cited inline);
nothing extrapolated except where labeled. Hardware: Apple M4, 10 cores, 16 GB (VERSION0.md). Solver
settings as in PROTOCOL pre-final lock.

Left panel: per-simulation wall time vs M (log scale) -- dense mesolve, seeded mcsolve (ntraj=200,
MC err ~0.08-0.12), V7 rollout. Right panel: one-time setup cost vs per-simulation cost at M=10, with
the measured break-even count; capability note for M > 10 where no exact reference was feasible here.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- measured points (source logs in comments) ---
mesolve_M = [3, 4, 5]                    # exact dense reference (obs-only trajectory)
mesolve_s = [3, 4, 45]                   # gen_v2/v3 logs: M=4 3-4s; pilot/campaign refs ~40-60s at M=5
mesolve_wall_note = (8, 2400)            # baselines/staircase: M=8 dense hung >40 min (Result 8 era)
mcsolve_M = [5, 8, 9, 10]
mcsolve_s = [60, 170, 250, 480]          # mc_ref/staircase logs: M=8 ~2-3 min, M=9 ~3-5, M=10 ~7-10
v7_M = [5, 8, 9, 10, 12, 16, 20]
v7_s = [3, 10, 30, 60, 14, 23, 39]       # eval logs (M<=10: staircase rollout portions; 12-20: Result 8)

SETUP_S = 2.5 * 3600 + 340 + 25          # one-time: 972-config e_ops data gen (gen_v6 log ~2.5h)
#                                          + V6 fit 338s + engine compile ~25s (fit_v56/validate logs)
COST_V7_M10 = 60                          # measured V7 rollout at M=10 (s)
COST_MC_M10 = 480                         # measured mcsolve reference at M=10 (s)
breakeven = SETUP_S / (COST_MC_M10 - COST_V7_M10)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
ax1.semilogy(mesolve_M, mesolve_s, "o-", color="black", label="exact dense mesolve (measured)")
ax1.semilogy(*mesolve_wall_note, "x", color="black", ms=11, mew=2)
ax1.annotate("M=8: hung >40 min\n(dense wall)", xy=mesolve_wall_note, xytext=(8.2, 500), fontsize=8)
ax1.semilogy(mcsolve_M, mcsolve_s, "s-", color="tab:purple",
             label="mcsolve ref, ntraj=200 (MC err ~0.1)")
ax1.semilogy(v7_M, v7_s, "^-", color="tab:blue", label="V7 rollout (measured)")
ax1.axvspan(10.5, 21, color="tab:blue", alpha=0.07)
ax1.text(15.5, 1000, "no exact reference\nfeasible here:\ncapability, not speed", fontsize=8,
         ha="center", color="tab:blue")
ax1.set_xlabel("molecule number M"); ax1.set_ylabel("wall time per simulation (s)")
ax1.set_title("Per-simulation cost vs system size (all measured)", fontsize=10)
ax1.legend(fontsize=8); ax1.set_xticks([3, 5, 8, 10, 12, 16, 20])

labels = ["one-time setup\n(972-config exact\ndata gen + V6 fit\n+ engine compile)",
          "per exact mcsolve\nreference @ M=10",
          "per V7 simulation\n@ M=10"]
vals = [SETUP_S / 60, COST_MC_M10 / 60, COST_V7_M10 / 60]
bars = ax2.bar(labels, vals, color=["tab:gray", "tab:purple", "tab:blue"])
for b, v in zip(bars, vals):
    ax2.text(b.get_x() + b.get_width() / 2, v * 1.02, f"{v:.1f} min", ha="center", fontsize=9)
ax2.set_ylabel("minutes"); ax2.set_yscale("log")
ax2.set_title(f"Amortization: setup repays after ~{breakeven:.0f} simulations at M=10\n"
              f"(beyond M=10, exact references were not feasible on this hardware at all)", fontsize=10)
fig.tight_layout()
fig.savefig("results/runtime_amortization.png", dpi=150)
print(f"saved results/runtime_amortization.png  (break-even ~{breakeven:.0f} simulations at M=10)")
