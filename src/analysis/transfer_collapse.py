"""
Why should a closure fitted on two or three emitters say anything about ten?

The design answer is that the estimator never sees M. It predicts each connected third cumulant
from (i) the low-order moments among that cumulant's own operators, (ii) the site detuning, and
(iii) an INTENSIVE context -- photon number per emitter, mean inversion, mean coherence. Nothing
in that list grows with system size, and one shared estimator is used for every emitter and every
M, so the fitted object is a function on a fixed feature space, not a function of M.

That is an argument. This script turns it into a measurement.

THE TEST
--------
If the map (local intensive features) -> kappa3 really is M-independent, then exact data taken at
different M must fall on ONE surface when plotted in those features. If instead each M traces its
own curve, the map depends on system size and small-M training cannot be expected to transfer --
and we should say so.

We take exact solves at several M, and for every emitter and every time step record

    features:  n/M (photons per emitter),  mean inversion zbar,  that emitter's detuning
    target  :  kappa3_i = <a^d a sz_i>_c            (the channel that enters the burst)

then ask whether kappa3_i, made intensive, is a single-valued function of the features across M.

This is a falsifiable check with a clear failure mode, not a plot that can only look reassuring:
a collapse supports transfer, a fan of separated curves refutes it.

Run:  python src/analysis/transfer_collapse.py [--smoke]
"""
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from kappa3_trace import solve_cell, kappa3_photon_inversion, NCS, WC  # noqa: E402


def features_and_target(S):
    """Per-(time, emitter) samples in the estimator's ACTUAL feature space.

    closure_v3.feature_plan/extract builds, for the monomial (a^d, a, sz_i):
        singles : <a^d>, <a>, <sz_i>            (all zero for the first two here, since <a>=0)
        pairs   : <a^d a>, <a^d sz_i>, <a sz_i>
        site    : the detuning of emitter i
        context : n/M, mean inversion, mean coherence          (intensive)
    Note that <a^d a> enters RAW and is extensive. So it is not true that every input is
    intensive; the honest claim is narrower, and is exactly what this script tests: whether the
    MAP from these features to kappa3 is the same function at every M. Binning in n/M instead of
    the real features would test a space the model never sees.
    """
    M = S["M"]
    k3 = kappa3_photon_inversion(S)               # (NT, M), complex
    ones = np.ones((1, M))
    npair = S["n"][:, None] * ones                # <a^d a>, the raw pair moment (extensive)
    nper = (S["n"] / M)[:, None] * ones           # intensive context entry
    zbar = S["sz"].mean(axis=1)[:, None] * ones
    zi = S["sz"]
    det = (S["w"] - WC)[None, :] * np.ones((S["n"].size, 1))
    return dict(
        npair=npair.ravel(),
        nper=nper.ravel(),
        zbar=zbar.ravel(),
        zi=zi.ravel(),
        adsz=np.real(S["adsz"]).ravel(),
        det=det.ravel(),
        k3=np.real(k3).ravel(),
        M=np.full(npair.size, M),
    )


def collapse_residual(data, nb=14, xkey="npair", ykey="zi"):
    """How well does a single M-independent surface fit all the M at once?

    We bin in (photons-per-emitter, mean inversion). Inside each bin, compare the spread of
    kappa3 ACROSS different M to the spread WITHIN a single M. If the map is M-independent the
    across-M spread should be no larger than the within-M noise, so the ratio sits near 1.
    Values much greater than 1 mean the curves separate by M, which would refute transfer.
    """
    x, y, k, Ms = data[xkey], data[ykey], data["k3"], data["M"]
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(k)
    x, y, k, Ms = x[keep], y[keep], k[keep], Ms[keep]
    xb = np.linspace(x.min(), x.max(), nb + 1)
    yb = np.linspace(y.min(), y.max(), nb + 1)
    ix = np.clip(np.digitize(x, xb) - 1, 0, nb - 1)
    iy = np.clip(np.digitize(y, yb) - 1, 0, nb - 1)

    across, within, counts = [], [], 0
    for a in range(nb):
        for b in range(nb):
            m = (ix == a) & (iy == b)
            if m.sum() < 12:
                continue
            present = np.unique(Ms[m])
            if len(present) < 2:
                continue
            means = [k[m & (Ms == q)].mean() for q in present]
            spreads = [k[m & (Ms == q)].std() for q in present]
            across.append(np.std(means))
            within.append(np.mean(spreads))
            counts += 1
    if not across:
        return np.nan, 0
    across, within = np.array(across), np.array(within)
    scale = max(np.abs(k).std(), 1e-12)
    return float(np.mean(across) / max(np.mean(within), 1e-12 * scale)), counts


def main():
    smoke = "--smoke" in sys.argv
    Ms = [2, 3] if smoke else [2, 3, 4, 5]
    tmax, nt = (20.0, 201) if smoke else (40.0, 401)

    print("Transfer collapse test: is kappa3 a single M-independent function of intensive features?")
    print()
    keys = ("npair", "nper", "zbar", "zi", "adsz", "det", "k3", "M")
    merged = {k: [] for k in keys}
    per_M = {}
    for M in Ms:
        S = solve_cell(M, NCS[M], tmax=tmax, nt=nt)
        d = features_and_target(S)
        per_M[M] = d
        for k in merged:
            merged[k].append(d[k])
        print(f"  M={M}: {d['k3'].size} (time, emitter) samples, "
              f"|kappa3| max {np.abs(d['k3']).max():.4f}, "
              f"photons/emitter max {d['nper'].max():.4f}")
    merged = {k: np.concatenate(v) for k, v in merged.items()}

    print()
    print("  binning coordinates                      across-M / within-M spread   bins")
    trials = [
        ("estimator features  (<a^d a>, <sz_i>)", "npair", "zi"),
        ("intensive context   (n/M,  zbar)     ", "nper", "zbar"),
        ("mixed               (<a^d a>, zbar)  ", "npair", "zbar"),
    ]
    results = {}
    for label, xk, yk in trials:
        r, nb = collapse_residual(merged, xkey=xk, ykey=yk)
        results[xk + "_" + yk] = r
        print(f"  {label}   {r:>10.3f}   {nb:>10d}")
    ratio, nbins = collapse_residual(merged, xkey="npair", ykey="zi")
    print()
    print("     ~1 or below  -> one M-independent surface describes every M: transfer is justified")
    print("     >> 1         -> the map depends on M and small-M training should not transfer")

    os.makedirs("results", exist_ok=True)
    np.savez_compressed("results/transfer_collapse.npz",
                        ratio=ratio, nbins=nbins,
                        **{f"{k}": merged[k] for k in merged})
    print("  saved results/transfer_collapse.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
