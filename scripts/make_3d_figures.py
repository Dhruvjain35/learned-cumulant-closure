"""Three-dimensional views of the exact dynamics and of the held-out test, light and dark. No new simulation.

    python run.py scripts/make_3d_figures.py

Inputs:  results/quantum_states_H11.npz    exact P(n, t) and per-atom excitation (scripts/export_quantum_states.py)
         results/heldout_param_burst.npz   the 30 pre-registered held-out cells (Result 40)
         results/heldout_param_cells.json  their parameters
         src/evaluation/training_support.py  the training grid
Outputs: assets/img/wigner-time{,-dark}.png   W(x, p = 0, t): a slice of the cavity's Wigner function through time
         assets/img/landscape{,-dark}.png     P(n, t) as 3D ridges, and each atom's excitation as a ribbon
         assets/img/paramspace{,-dark}.png    the held-out cells and the training grid in (kappa, gamma_phi, g_c)
"""
import json
import numpy as np
import qutip as qt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, LightSource, TwoSlopeNorm, Normalize
from mpl_toolkits.mplot3d.art3d import PolyCollection

for f in ("/System/Library/Fonts/HelveticaNeue.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf"):
    try:
        font_manager.fontManager.addfont(f)
    except Exception:
        pass

THEMES = {
    "": dict(bg="#ffffff", fg="#0d0d0d", sub="#5b5b5b", pane=(0.96, 0.96, 0.97, 1), edge="#d0d0d0",
             wig=["#c2410c", "#fb923c", "#d6e2f2", "#3b82f6", "#1e3a8a"], v7="#1d6ff2", plain="#e8590c",
             train="#9aa0a6", ramp=["#dbe8fd", "#1d6ff2", "#0b2f7a"], atom=["#f59e0b", "#b45309"]),
    "-dark": dict(bg="#0d1117", fg="#f0f3f6", sub="#9aa4ae", pane=(0.08, 0.10, 0.13, 1), edge="#30363d",
                  wig=["#ff8a3d", "#c2410c", "#1b2230", "#1e4fb8", "#7cb8ff"], v7="#4c9aff", plain="#ff7b3a",
                  train="#6e7681", ramp=["#1d3a66", "#3b82f6", "#bfdbfe"], atom=["#fbbf24", "#f97316"]),
}


def style(c):
    plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 13,
                         "figure.facecolor": c["bg"], "savefig.facecolor": c["bg"], "axes.facecolor": c["bg"],
                         "text.color": c["fg"], "axes.labelcolor": c["sub"], "xtick.color": c["sub"],
                         "ytick.color": c["sub"], "axes.edgecolor": c["edge"]})


def panes(ax, c):
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color(c["pane"])
        axis._axinfo["grid"].update(color=c["edge"], linewidth=0.6)
        axis.line.set_color(c["edge"])
    ax.set_facecolor(c["bg"])
    ax.tick_params(pad=2, labelsize=11)


def wigner_time(z, sfx, c, cache={}):
    ts, P = z["t"], z["P"]
    sel = ts <= 12.0 + 1e-9
    ts, P = ts[sel][::2], P[sel][::2]
    xv = np.linspace(-3.4, 3.4, 121)
    if "W" not in cache:
        cache["W"] = np.array([qt.wigner(qt.Qobj(np.diag(p)), xv, np.array([0.0]))[0] for p in P])
    W = cache["W"]
    fig = plt.figure(figsize=(12.5, 6.4))
    ax = fig.add_axes([-0.06, -0.06, 1.12, 1.1], projection="3d")
    X, T = np.meshgrid(xv, ts)
    wcm = LinearSegmentedColormap.from_list("w", c["wig"])
    norm = TwoSlopeNorm(vmin=min(W.min(), -1e-3), vcenter=0, vmax=W.max())
    rgb = LightSource(azdeg=300, altdeg=40).shade(W, cmap=wcm, norm=norm, blend_mode="soft", vert_exag=8)
    ax.plot_surface(T, X, W, facecolors=rgb, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
    ax.set_xlabel("time", labelpad=10)
    ax.set_ylabel("field quadrature x", labelpad=10)
    ax.set_zlabel("W(x, p = 0)", labelpad=8)
    ax.view_init(elev=30, azim=-62)
    ax.set_box_aspect((2.3, 1.15, 0.62))
    panes(ax, c)
    k = int(np.argmin(W[:, len(xv) // 2]))
    fig.text(0.05, 0.93, "The cavity's Wigner function through time (exact, held-out system H11)", fontsize=15,
             fontweight="bold", color=c["fg"])
    wtxt = f"{W.min():+.3f}".replace("-", "\u2212")
    fig.text(0.05, 0.89, f"A slice through phase space at p = 0. Orange regions are negative, which classical light "
                         f"cannot be. Deepest point: W = {wtxt} at t = {ts[k]:.1f}, the burst peak.", fontsize=12.5,
             color=c["sub"])
    fig.savefig(f"assets/img/wigner-time{sfx}.png", dpi=150)
    plt.close(fig)
    return W.min(), ts[k]


def landscape(z, sfx, c):
    ts, P, pe = z["t"], z["P"], z["p_exc"]
    det = z["detunings"]
    sel = ts <= 15.0 + 1e-9
    ts, P, pe = ts[sel], P[sel], pe[sel]
    fig = plt.figure(figsize=(13, 5.6))
    # (a) photon-number ridges
    ax = fig.add_axes([-0.04, 0.0, 0.56, 0.92], projection="3d")
    ramp = LinearSegmentedColormap.from_list("r", c["ramp"])
    nmax = 6
    verts, cols = [], []
    for n in range(nmax):
        y = P[:, n]
        verts.append([(ts[0], 0)] + list(zip(ts, y)) + [(ts[-1], 0)])
        cols.append(ramp(0.15 + 0.85 * n / (nmax - 1)))
    pc = PolyCollection(verts, facecolors=cols, edgecolors=[c["bg"]] * nmax, linewidths=1.2, alpha=0.92)
    ax.add_collection3d(pc, zs=np.arange(nmax), zdir="y")
    ax.set_xlim(0, 15); ax.set_ylim(-0.5, nmax - 0.5); ax.set_zlim(0, 1)
    ax.set_xlabel("time", labelpad=8); ax.set_ylabel("photons n", labelpad=6); ax.set_zlabel("P(n, t)", labelpad=6)
    ax.set_yticks(range(nmax))
    ax.view_init(elev=26, azim=-58)
    ax.set_box_aspect((2.0, 1.25, 0.75))
    panes(ax, c)
    fig.text(0.04, 0.92, "Photon-number distribution", fontsize=14.5, fontweight="bold", color=c["fg"])
    fig.text(0.04, 0.875, "Probability of n photons in the cavity over time", fontsize=12, color=c["sub"])

    # (b) per-atom excitation ribbons, ordered by detuning
    ax2 = fig.add_axes([0.47, 0.0, 0.56, 0.92], projection="3d")
    order = np.argsort(det)
    acm = LinearSegmentedColormap.from_list("a", c["atom"])
    span = max(abs(det).max(), 1e-9)
    for j, i in enumerate(order):
        y = pe[:, i]
        ax2.add_collection3d(PolyCollection([[(ts[0], 0)] + list(zip(ts, y)) + [(ts[-1], 0)]],
                                            facecolors=[acm(0.5 + 0.5 * det[i] / span)], edgecolors=[c["bg"]],
                                            linewidths=1.2, alpha=0.9), zs=j, zdir="y")
    ax2.set_xlim(0, 15); ax2.set_ylim(-0.5, len(order) - 0.5); ax2.set_zlim(0, 1)
    ax2.set_yticks(range(len(order)))
    ax2.set_yticklabels([f"{det[i]:+.2f}".replace("-", "−") for i in order], fontsize=10)
    ax2.set_xlabel("time", labelpad=8); ax2.set_ylabel("atom detuning", labelpad=12); ax2.set_zlabel("excitation", labelpad=6)
    ax2.view_init(elev=26, azim=-58)
    ax2.set_box_aspect((2.0, 1.25, 0.75))
    panes(ax2, c)
    fig.text(0.53, 0.92, "Each atom's excitation", fontsize=14.5, fontweight="bold", color=c["fg"])
    fig.text(0.53, 0.875, "Atoms tuned away from the cavity give up their energy differently", fontsize=12, color=c["sub"])
    fig.savefig(f"assets/img/landscape{sfx}.png", dpi=150)
    plt.close(fig)


def paramspace(sfx, c):
    rows = {r["id"]: r for r in np.load("results/heldout_param_burst.npz", allow_pickle=True)["rows"]}
    cells = json.load(open("results/heldout_param_cells.json"))["cells"]
    import training_support as tsup
    grid = np.unique(np.asarray(tsup.TRAIN)[:, :3], axis=0)   # (g_c, kappa, gamma_phi); disorder levels collapse
    fig = plt.figure(figsize=(12.5, 6.6))
    ax = fig.add_axes([0.0, -0.04, 0.86, 1.06], projection="3d")
    lk = lambda v: np.log10(v)
    if grid is not None:
        g = np.array([(lk(p[1]), p[2], p[0]) for p in grid])
        ax.scatter(g[:, 0], g[:, 1], g[:, 2], s=26, color=c["train"], alpha=0.55, depthshade=False, label="training")
    err = np.array([100 * rows[cl["id"]]["v7"]["abs"] for cl in cells])
    P = np.array([(lk(cl["kappa"]), cl["gamma_phi"], cl["g_c"]) for cl in cells])
    sig = np.array([cl["sigma"] for cl in cells])
    ramp = LinearSegmentedColormap.from_list("r", c["ramp"])
    zfloor = 0.25
    for (x, y, zz) in P:
        ax.plot([x, x], [y, y], [zfloor, zz], color=c["edge"], lw=0.9)
    sc = ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=60 + 900 * sig ** 1.2, c=err, cmap=ramp,
                    norm=Normalize(0, 1.5), edgecolor=c["fg"], linewidth=0.6, depthshade=False, label="held-out")
    ax.scatter(P[:, 0], P[:, 1], np.full(len(P), zfloor), s=12, color=c["edge"], depthshade=False)
    ax.set_xlabel("cavity loss κ", labelpad=10)
    ax.set_ylabel("dephasing γφ", labelpad=10)
    ax.set_zlabel("coupling g", labelpad=8)
    xt = [0.02, 0.05, 0.1, 0.2, 0.4]
    ax.set_xticks([lk(v) for v in xt]); ax.set_xticklabels([f"{v:g}" for v in xt])
    ax.set_zlim(zfloor, 1.25)
    ax.view_init(elev=22, azim=-52)
    ax.set_box_aspect((1.6, 1.2, 0.9))
    panes(ax, c)
    cax = fig.add_axes([0.86, 0.22, 0.015, 0.5])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("learned-closure error (%)", color=c["sub"])
    cb.outline.set_edgecolor(c["edge"])
    fig.text(0.05, 0.93, "The 30 held-out systems in parameter space", fontsize=15, fontweight="bold", color=c["fg"])
    fig.text(0.05, 0.905, "Colour is the learned closure's burst-peak error and size is the disorder strength; grey dots are "
                          "the training grid.\nThe standard method is 28.5–33.1% off at every one of these points.",
             fontsize=12.5, color=c["sub"], linespacing=1.5, va="top")
    fig.savefig(f"assets/img/paramspace{sfx}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    z = np.load("results/quantum_states_H11.npz")
    for sfx, c in THEMES.items():
        style(c)
        wmin, tmin = wigner_time(z, sfx, c)
        landscape(z, sfx, c)
        paramspace(sfx, c)
    print(f"wrote wigner-time, landscape, paramspace (light + dark); min W(x,0,t) = {wmin:+.4f} at t = {tmin:.2f}")
