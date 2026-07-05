"""
PLOT ONLY. Reads the frozen, already-exported trajectory file
results/baseline_burst_trace_M3_pocket_draw50.npz and renders the baseline burst-trace figure (PNG + PDF).

This script does NOT run, rebuild, or alter any model. It reads saved trajectory arrays only.
Regenerate the input with:  python run.py scripts/export_baseline_burst_trace.py

Run:  python run.py scripts/plot_baseline_burst_trace.py   (only needs numpy + matplotlib)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "results/baseline_burst_trace_M3_pocket_draw50.npz"
d = np.load(SRC, allow_pickle=True)
m = d["meta"].item()

fig, ax = plt.subplots(figsize=(8.2, 5.0))
ax.plot(d["ts_exact"], d["n_exact"], color="black",  lw=2.6, label="exact (master equation)", zorder=6)
ax.plot(d["ts_v7"],    d["n_v7"],    color="#2a9d8f", lw=1.9, ls="-",  label="V7 (learned)", zorder=5)
ax.plot(d["ts_plain"], d["n_plain"], color="#e9a23b", lw=1.7, ls="--", label="2nd-order cumulant")
ax.plot(d["ts_t3"],    d["n_t3"],    color="#e76f51", lw=1.7, ls=":",  label=r"$\kappa_4=0$ (3rd order)")
ax.plot(d["ts_t4"],    d["n_t4"],    color="#9b2226", lw=1.7, ls="-.", label=r"$\kappa_5=0$ (4th order)")

ax.set_ylim(-0.6, 4.6)
ax.set_xlim(0, 40)

# mark the kappa5=0 (T4) truncation at the divergence guard
if bool(m.get("t4_guard_triggered", False)):
    tstop = float(m["t4_stop"])
    ax.axvline(tstop, color="#9b2226", lw=1.0, ls=":", alpha=0.8)
    ax.annotate(rf"$\kappa_5=0$ trace truncated at the" + "\n" + rf"divergence guard ($t\approx{tstop:.2f}$)",
                xy=(tstop, 3.4), xytext=(tstop + 2.2, 4.1), ha="left", fontsize=8.5, color="#9b2226",
                arrowprops=dict(arrowstyle="->", color="#9b2226", lw=0.9))

ax.set_xlabel("time  $t$")
ax.set_ylabel(r"cavity occupation  $\langle a^\dagger a\rangle(t)$")
ax.set_title(f"Superradiant burst  —  M={m['M']}, {m['regime']}, disorder draw {m['seed']}", fontsize=10.5)
ax.grid(alpha=0.25)
ax.legend(fontsize=8.5, loc="upper right", framealpha=0.95)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"results/baseline_burst_trace_M3_pocket_draw50.{ext}", dpi=150)
print("wrote results/baseline_burst_trace_M3_pocket_draw50.png and .pdf")
