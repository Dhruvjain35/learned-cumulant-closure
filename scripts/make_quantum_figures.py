"""'Inside the burst': the exact photon-number distribution of the cavity over time, and its Wigner function.

    python run.py scripts/make_quantum_figures.py

Inputs:  results/quantum_states_H11.npz (exact P(n, t), from scripts/export_quantum_states.py)
         demo/data/heldout.json         (2nd-order and learned-closure <a+a>(t) for the same cell)
Output:  assets/img/inside-burst{,-dark}.png

The cavity's reduced state is diagonal in the Fock basis in this model (see export_quantum_states.py), so the Wigner
function at each time is W(x, p) = sum_n P(n, t) W_n(x, p): rotationally symmetric, with negative regions wherever
odd photon numbers dominate near the origin.
"""
import json
import numpy as np
import qutip as qt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, LightSource, TwoSlopeNorm

for f in ("/System/Library/Fonts/HelveticaNeue.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf"):
    try:
        font_manager.fontManager.addfont(f)
    except Exception:
        pass

THEMES = {
    "": dict(bg="#ffffff", fg="#0d0d0d", sub="#5b5b5b", grid="#e9e9e9", mean="#0d0d0d", plain="#e8590c",
             v7="#1d6ff2", heat=["#ffffff", "#dbe8fd", "#86b3fa", "#1d6ff2", "#0b2f7a"],
             wig=["#ea580c", "#fb923c", "#eef2f7", "#7fb0f8", "#1d4ed8"]),
    "-dark": dict(bg="#0d1117", fg="#f0f3f6", sub="#9aa4ae", grid="#21262d", mean="#f0f3f6", plain="#ff7b3a",
                  v7="#4c9aff", heat=["#0d1117", "#10294d", "#1d5fc4", "#5aa2ff", "#e6f0ff"],
                  wig=["#ff8a3d", "#c2410c", "#1b2230", "#1e4fb8", "#7cb8ff"]),
}


def main():
    z = np.load("results/quantum_states_H11.npz")
    ts, P, nt = z["t"], z["P"], z["n"]
    cell = next(c for c in json.load(open("demo/data/heldout.json"))["cells"] if c["id"] == "H11")
    tc = np.array(cell["t"])
    nmaxph = 6
    xv = np.linspace(-3.6, 3.6, 161)
    ipk = int(np.argmax(nt))
    picks = [(int(np.argmin(np.abs(ts - 1.6))), "rising"), (ipk, "burst peak"), (int(np.argmin(np.abs(ts - 7.0))), "after the burst")]
    Ws = [qt.wigner(qt.Qobj(np.diag(P[i])), xv, xv) for i, _ in picks]
    wlim = max(abs(W).max() for W in Ws)

    for sfx, c in THEMES.items():
        plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 14,
                             "figure.facecolor": c["bg"], "savefig.facecolor": c["bg"], "axes.facecolor": c["bg"],
                             "text.color": c["fg"], "axes.labelcolor": c["sub"], "xtick.color": c["sub"],
                             "ytick.color": c["sub"], "axes.edgecolor": c["sub"]})
        fig = plt.figure(figsize=(12.5, 10.4))
        ax = fig.add_axes([0.08, 0.60, 0.86, 0.35])
        heat = LinearSegmentedColormap.from_list("h", c["heat"])
        ax.imshow(P[:, :nmaxph].T, origin="lower", aspect="auto", cmap=heat, vmin=0, vmax=0.55,
                  extent=[ts[0], ts[-1], -0.5, nmaxph - 0.5], interpolation="nearest")
        for k in range(nmaxph + 1):
            ax.axhline(k - 0.5, color=c["bg"], lw=2.2)
        ax.plot(ts, nt, color=c["mean"], lw=2.6, label="exact mean")
        ax.plot(tc, cell["plain"]["n"], color=c["plain"], lw=2.2, ls=(0, (6, 3)), label="2nd-order closure")
        ax.plot(tc, cell["v7"]["n"], color=c["v7"], lw=2.0, ls=(0, (1.5, 1.5)), label="learned closure")
        ax.set_xlim(0, 15)
        ax.set_ylim(-0.5, nmaxph - 0.5)
        ax.set_yticks(range(nmaxph))
        ax.set_xlabel("time")
        ax.set_ylabel("photons n")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=0, pad=8)
        lg = ax.legend(loc="upper right", frameon=False, fontsize=13)
        for t, l in zip(lg.get_texts(), lg.get_lines()):
            t.set_color(l.get_color())
        ax.text(0.0, 1.04, "Photon-number distribution P(n, t) inside the cavity, exact (5 atoms, held-out system H11)",
                transform=ax.transAxes, color=c["fg"], fontsize=15, fontweight="bold")
        for i, _ in picks:
            ax.axvline(ts[i], color=c["sub"], lw=1, ls=":", alpha=.8)

        wcm = LinearSegmentedColormap.from_list("w", c["wig"])
        wneg = min(min(W.min() for W in Ws), -1e-3)
        norm = TwoSlopeNorm(vmin=wneg, vcenter=0, vmax=wlim)
        ls = LightSource(azdeg=315, altdeg=38)
        X, Y = np.meshgrid(xv, xv)
        for k, ((i, name), W) in enumerate(zip(picks, Ws)):
            a3 = fig.add_axes([-0.01 + k * 0.33, -0.05, 0.36, 0.50], projection="3d")
            a3.set_facecolor(c["bg"])
            rgb = ls.shade(W, cmap=wcm, norm=norm, blend_mode="soft", vert_exag=10)
            a3.plot_surface(X, Y, W, facecolors=rgb, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
            a3.set_zlim(-0.55 * wlim, wlim)
            a3.view_init(elev=52, azim=-58)
            a3.set_box_aspect((1, 1, 0.62), zoom=1.25)
            for axis in (a3.xaxis, a3.yaxis, a3.zaxis):
                axis.set_pane_color((0, 0, 0, 0))
                axis.line.set_color((0, 0, 0, 0))
            a3.set_xticks([]); a3.set_yticks([]); a3.set_zticks([])
            a3.grid(False)
            w0 = float(W[len(xv) // 2, len(xv) // 2])
            nmean = float(P[i] @ np.arange(P.shape[1]))
            fig.text(0.17 + k * 0.33, 0.44, f"t = {ts[i]:.1f}, {name}", color=c["fg"], fontsize=14,
                     fontweight="bold", ha="center")
            fig.text(0.17 + k * 0.33, 0.415, f"⟨n⟩ = {nmean:.2f}     W(0,0) = {w0:+.3f}".replace("-", "\u2212"),
                     color=(c["plain"] if w0 < 0 else c["sub"]), fontsize=13, ha="center")
        fig.text(0.08, 0.49, "Wigner function of the cavity light at the three dotted times. Orange is negative, which no"
                             " classical light can do.", color=c["sub"], fontsize=13.5)
        fig.savefig(f"assets/img/inside-burst{sfx}.png", dpi=150)
        plt.close(fig)
        print(f"wrote assets/img/inside-burst{sfx}.png")
    for (i, name), W in zip(picks, Ws):
        print(f"{name:>16s}: t={ts[i]:.2f}  W(0,0)={W[80, 80]:+.4f}  min W={W.min():+.4f}")


if __name__ == "__main__":
    main()
