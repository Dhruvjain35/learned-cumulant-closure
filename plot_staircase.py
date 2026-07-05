"""Staircase figure (review spec): one figure showing, vs molecule number M --
   (top)  NRMSE-when-stable per seed (markers) + per-M mean of stable seeds (line), with the MC reference
          uncertainty band, for plain / v2 / v3-RATIO;
   (bottom) stability rate (of 3 seeds) per method.
So a good average cannot hide rare blowups, and a noisy MC reference cannot hide (or fake) a real win."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.load("results/staircase.npz", allow_pickle=True)["rows"].item()
MS = sorted({k[0] for k in d})
SEEDS = sorted({k[1] for k in d})
METH = [("plain", "tab:red"), ("v2", "tab:orange"), ("v3rat", "tab:blue")]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 7), sharex=True,
                               gridspec_kw={"height_ratios": [3, 1]})
# MC reference uncertainty band (normalized: MCerr is abs; NRMSE denominator ~ RMS(n_ref); approximate
# band as mean MCerr / mean peak -- labeled as approximate)
mc = [np.mean([d[(M, s)][1] for s in SEEDS]) for M in MS]
ax1.fill_between(MS, 0, np.array(mc) / 2.0, color="gray", alpha=0.25,
                 label="~MC reference uncertainty (approx. normalized)")
for name, c in METH:
    means = []
    for M in MS:
        vals = [d[(M, s)][0][name]["nn"] for s in SEEDS if d[(M, s)][0][name]["ok"]]
        for s in SEEDS:
            r = d[(M, s)][0][name]
            if r["ok"]:
                ax1.plot(M, r["nn"], "o", color=c, ms=4, alpha=0.55)
            else:
                ax1.plot(M, 0.62, "x", color=c, ms=9, mew=2)      # blowup marker at top
        means.append(np.mean(vals) if vals else np.nan)
    ax1.plot(MS, means, "-", color=c, lw=1.8, label=name)
ax1.set_ylabel("NRMSE (photon number), stable runs")
ax1.set_ylim(0, 0.65)
ax1.legend(fontsize=8)
ax1.set_title("Size-transfer staircase at the V0 regime, seeds 7/8/9, seeded mcsolve references\n"
              "(x at top = unstable rollout; per-seed dots + stable-mean line)", fontsize=10)
for name, c in METH:
    rate = [np.mean([d[(M, s)][0][name]["ok"] for s in SEEDS]) for M in MS]
    ax2.plot(MS, rate, "-o", color=c, ms=4, lw=1.6)
ax2.set_ylim(-0.05, 1.05)
ax2.set_ylabel("stability rate")
ax2.set_xlabel("molecule number M (training used M <= 5)")
ax2.set_xticks(MS)
fig.tight_layout()
fig.savefig("results/staircase.png", dpi=150)
print("saved results/staircase.png")
