"""
Can cheap small-M cells cover the feature range that expensive large-M cells occupy?

THE IDEA
--------
scaling_search.py showed the bind: kappa3 is nearly a single M-independent function of the raw
features (collapse 1.12), but <a^d a> is extensive, so a large system sits outside the window a
small system defines -- 41.5% overlap at the burst peak, which is exactly where the claim lives.
No rescaling fixes both at once.

But if the map really is M-independent in (<a^d a>, <sz_i>), then HOW a cell arrives at a given
<a^d a> should not matter. Large <a^d a> is reachable at small M by other means: raise the
collective coupling, lower the cavity loss, lower the dephasing. Those cells are cheap -- M=2,3
exact solves run in seconds.

If kappa3 at matched (<a^d a>, <sz_i>) agrees between
    (a) small M, strong coupling / low loss        [cheap]
    (b) larger M, standard coupling                [expensive]
then the training set can be extended to cover the large-M feature range WITHOUT large-M exact
solves, and size-extrapolation becomes interpolation. That would be a real route to larger M.

If they disagree, the M-dependence is genuine and irreducible, no amount of clever sampling helps,
and the ceiling should be reported as physics.

This is a falsifiable test with a real failure mode. Nothing is trained; it is a property of the
exact dynamics either way.

Run:  python src/analysis/reach_by_regime.py [--smoke]
"""
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "engines"))

import qutip as qt  # noqa: E402
from exact_tc import build_ops, build_H, collapse_ops, draw_disorder  # noqa: E402

WC, GDOWN, SIGMA, SEED = 1.0, 0.02, 0.15, 7


def solve(M, Nc, gc, kappa, gphi, tmax, nt, seed=SEED):
    """Exact solve at an arbitrary regime point; returns the (n, z_i, kappa3_i) samples."""
    w = draw_disorder(M, WC, SIGMA, seed, "gauss")
    g = np.full(M, gc / np.sqrt(M))
    ops = build_ops(M, Nc)
    H = build_H(ops, WC, w, g)
    c = collapse_ops(ops, kappa=kappa, gphi=gphi, gdown=GDOWN)
    psi0 = qt.tensor(qt.basis(Nc, 0), *[qt.basis(2, 0) for _ in range(M)])
    ts = np.linspace(0.0, tmax, nt)
    a, sz = ops["a"], ops["sz"]
    n_op = a.dag() * a
    e_ops = [n_op, a] + [n_op * sz[i] for i in range(M)] + [sz[i] for i in range(M)] \
        + [a * sz[i] for i in range(M)] + [a.dag() * sz[i] for i in range(M)]
    res = qt.mesolve(H, psi0, ts, c_ops=c, e_ops=e_ops,
                     options={"atol": 1e-11, "rtol": 1e-9, "nsteps": 20000})
    E = [np.asarray(x) for x in res.expect]
    o = 0
    n = E[o].real; o += 1
    aex = E[o]; o += 1
    adasz = np.stack(E[o:o + M], axis=1); o += M
    szs = np.stack(E[o:o + M], axis=1).real; o += M
    asz = np.stack(E[o:o + M], axis=1); o += M
    adsz = np.stack(E[o:o + M], axis=1); o += M
    ad, aa = np.conj(aex)[:, None], aex[:, None]
    k3 = np.real(adasz - ad * asz - aa * adsz - szs * n[:, None] + 2.0 * ad * aa * szs)
    ones = np.ones((1, M))
    return dict(n=(n[:, None] * ones).ravel(), zi=szs.ravel(), k3=k3.ravel(),
                peak_n=float(n.max()), M=M, gc=gc, kappa=kappa, gphi=gphi)


def matched_disagreement(cheap, ref, nb=12, minn=10):
    """Median |k3_cheap - k3_ref| in shared (n, z) bins, relative to the local |k3| scale.

    Restricted to the region the reference actually reaches, since that is what we are trying to
    cover. Returns (relative disagreement, number of shared bins, coverage of ref by cheap).
    """
    lo_x, hi_x = ref["n"].min(), ref["n"].max()
    lo_y, hi_y = ref["zi"].min(), ref["zi"].max()
    xb = np.linspace(lo_x, hi_x, nb + 1)
    yb = np.linspace(lo_y, hi_y, nb + 1)

    def binned(d):
        ix = np.clip(np.digitize(d["n"], xb) - 1, 0, nb - 1)
        iy = np.clip(np.digitize(d["zi"], yb) - 1, 0, nb - 1)
        inside = (d["n"] >= lo_x) & (d["n"] <= hi_x) & (d["zi"] >= lo_y) & (d["zi"] <= hi_y)
        out = {}
        for a in range(nb):
            for b in range(nb):
                m = (ix == a) & (iy == b) & inside
                if m.sum() >= minn:
                    out[(a, b)] = d["k3"][m].mean()
        return out

    A, B = binned(cheap), binned(ref)
    shared = sorted(set(A) & set(B))
    if not shared:
        return np.nan, 0, 0.0
    da = np.array([A[k] for k in shared])
    db = np.array([B[k] for k in shared])
    scale = max(np.abs(db).mean(), 1e-12)
    return float(np.median(np.abs(da - db)) / scale), len(shared), len(shared) / max(len(B), 1)


def main():
    smoke = "--smoke" in sys.argv
    tmax, nt = (20.0, 201) if smoke else (30.0, 301)

    print("Can cheap small-M cells cover the large-M feature range, with the same kappa3?")
    print()

    # the expensive reference we are trying to imitate: standard regime at the largest M we solve
    ref = solve(5, 12, 0.7, 0.03, 0.02, tmax, nt)
    print(f"  reference   M=5 gc=0.70 kappa=0.030 : peak <a^d a> = {ref['peak_n']:.3f}")
    print()

    # cheap probes: few emitters, pushed to high photon number by regime instead of by size
    probes = [
        (2, 12, 1.10, 0.020, 0.02),
        (2, 14, 1.30, 0.015, 0.02),
        (3, 12, 1.00, 0.020, 0.02),
        (3, 14, 1.20, 0.015, 0.02),
        (2, 12, 0.70, 0.030, 0.02),      # control: same regime as the reference, small M
    ]
    print(f"  {'cheap cell':<34} {'peak n':>8} {'reach':>7} {'shared':>7} {'disagree':>9}")
    rows = []
    for (M, Nc, gc, ka, gp) in probes:
        d = solve(M, Nc, gc, ka, gp, tmax, nt)
        dis, nsh, cov = matched_disagreement(d, ref)
        tag = f"M={M} gc={gc:.2f} kappa={ka:.3f}"
        reach = d["peak_n"] / ref["peak_n"]
        rows.append((tag, d["peak_n"], reach, nsh, cov, dis))
        print(f"  {tag:<34} {d['peak_n']:>8.3f} {reach:>6.2f}x {nsh:>7d} "
              f"{100 * dis:>8.1f}%")

    print()
    print("  reach   = peak <a^d a> relative to the M=5 reference (>=1 means it gets there)")
    print("  shared  = number of (n, z) bins where both have data")
    print("  disagree= median |k3 mismatch| in those bins, as a % of the local |k3| scale")
    print()
    ok = [r for r in rows if r[2] >= 0.95 and r[5] < 0.15]
    if ok:
        print("  -> at least one cheap cell reaches the reference's range AND agrees on kappa3.")
        print("     Training can then be extended by REGIME instead of by size.")
    else:
        best = min((r for r in rows if r[2] >= 0.95), key=lambda r: r[5], default=None)
        if best is None:
            print("  -> no cheap cell reached the reference's photon range; push gc higher / kappa lower.")
        else:
            print(f"  -> cells reach the range, but kappa3 disagrees by {100*best[5]:.1f}% there.")
            print("     The cumulant is not a function of (n, z) alone; regime enters separately,")
            print("     so covering the range is necessary but not sufficient.")

    os.makedirs("results", exist_ok=True)
    np.savez_compressed("results/reach_by_regime.npz",
                        tags=np.array([r[0] for r in rows]),
                        peak_n=np.array([r[1] for r in rows]),
                        reach=np.array([r[2] for r in rows]),
                        shared=np.array([r[3] for r in rows]),
                        disagree=np.array([r[5] for r in rows]),
                        ref_peak=ref["peak_n"])
    print("  saved results/reach_by_regime.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
