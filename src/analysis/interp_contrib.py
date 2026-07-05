"""
Interpretability step 1 (PROTOCOL): rank quadratic terms by ACTUAL CONTRIBUTION along closed-loop
trajectories, c_j = w_j * phi_j(x_std), per region, for the dominant cumulant type -- Q21 and Q27 side by
side, grouped by physics family. Dev seeds 35/36 only.

Feature names (post-ratio-transform, the space the models actually see):
  0-5   s1r s1i s2r s2i s3r s3i        singles of the monomial's own factors
  6-11  p12r p12i p13r p13i p23r p23i  pairs
  12-14 d1/k d2/k d3/k                 site detunings / kappa
  15-17 nc/M zbar coh                  instantaneous collective context
  18-20 g/k k gphi/k                   regime
  21-26 h_nc dnc h_z dz h_coh dcoh     traces and current-minus-trace (Q27 only)
"""
import pickle
import numpy as np

from closure_v3 import ClosureApplyV3, ratio_transform, context_v3, canonical_sig, conj_key
from closure_v6 import ClosureApplyV6, run_v6, lambdas, trace_features
from local_closure import feature_plan, extract
from baseline_shell import LocalClosureShell  # noqa: F401
from pilot_eval import build_cell, m0_product, run_from, WC
from cumulant import mono_order

DEV_SEEDS = [35, 36]
LEDGER_TRAIN = {5, 15, 30, 105, 115, 130, 205, 215, 230}
assert not (set(DEV_SEEDS) & LEDGER_TRAIN)
GC, SG, M5 = 0.7, 0.15, 5
REGIONS = [("normal", 0.15, 0.02), ("box", 0.03, 0.02), ("seam", 0.05, 0.02), ("resid", 0.03, 0.065)]
NAMES21 = ["s1r", "s1i", "s2r", "s2i", "s3r", "s3i", "p12r", "p12i", "p13r", "p13i", "p23r", "p23i",
           "d1/k", "d2/k", "d3/k", "nc/M", "z", "coh", "g/k", "k", "gphi/k"]
NAMES27 = NAMES21 + ["h_nc", "dnc", "h_z", "dz", "h_coh", "dcoh"]
DOM = ("sp", "sm", "sz")                                  # the largest dynamically-dominant type


def family(term):
    if any(t in term for t in ("h_nc", "dnc", "h_z", "dz", "h_coh", "dcoh")):
        return "trace"
    if any(t in term for t in ("d1/k", "d2/k", "d3/k", "g/k", "gphi/k")) or term == "k" or " k" in f" {term}":
        return "regime"
    if " " in term:
        return "nl-product"
    return "instant"


def features_along(model_name, lc, eng, det, regime, m0):
    """Roll the model out; return the raw 21/27-col feature rows its dominant-type net saw (recomputed
    along the saved trajectory; traces from run_v6's own h_traj for Q27)."""
    keys = [k for k in eng.o3_keys if canonical_sig(k)[0] == DOM and not canonical_sig(k)[1]]
    plans = {k: feature_plan(conj_key(k) if canonical_sig(k)[1] else k) for k in keys}
    if model_name == "q27":
        _, traj, h_traj = run_v6(eng, m0, ClosureApplyV6(lc, eng, det, regime, xform=ratio_transform),
                                 lambdas(regime[1], regime[2]))
    else:
        _, traj = run_from(eng, m0, ClosureApplyV3(lc, eng, det, regime, xform=ratio_transform))
        h_traj = None
    rows = []
    for ti in range(0, len(traj), 4):
        m = traj[ti]
        mdict = {k: m[i] for i, k in enumerate(eng.track)}
        ctx = context_v3(mdict, eng.M)
        tr = list(trace_features(np.array(ctx), h_traj[ti])) if h_traj is not None else []
        for k in keys:
            rows.append(extract(mdict, plans[k], det, ctx) + list(regime) + tr)
    return np.array(rows)


def contributions(lc, X_raw, names):
    """c_j = w_j * phi_j over rows, for the dominant type's pipeline. Returns {term: (med, p95, signcons)}."""
    net = lc.nets[DOM]
    sc = lc.scalers[DOM]
    Z = sc.transform(ratio_transform(X_raw) if X_raw.shape[1] >= 21 else X_raw)
    poly = net.named_steps["poly"]
    Phi = poly.transform(Z)
    W = net.named_steps["lasso"].coef_            # (2, nterms) for [re, im]
    tnames = poly.get_feature_names_out(names)
    out = {}
    for j in range(Phi.shape[1]):
        w = W[:, j]
        if np.all(np.abs(w) < 1e-10):
            continue
        c = np.abs(Phi[:, j])[:, None] * np.abs(w)[None, :]
        cmag = np.max(c, axis=1)                  # worst-component contribution per row
        sgn = np.sign(Phi[:, j] * (w[0] if abs(w[0]) > abs(w[1]) else w[1]))
        cons = float(np.mean(sgn == np.sign(np.median(sgn)))) if np.any(sgn) else 0.0
        out[tnames[j]] = (float(np.median(cmag)), float(np.percentile(cmag, 95)), cons)
    return out


if __name__ == "__main__":
    with open("results/lc_q21.pkl", "rb") as fh:
        q21 = pickle.load(fh)
    with open("results/lc_q27.pkl", "rb") as fh:
        q27 = pickle.load(fh)
    print(f"contribution analysis, dominant type {DOM}, dev seeds {DEV_SEEDS}\n")
    tables = {}
    for (rtag, ka, gp) in REGIONS:
        for mdl, lc, names in (("q21", q21, NAMES21), ("q27", q27, NAMES27)):
            allrows = []
            for sd in DEV_SEEDS:
                eng, w, g = build_cell(M5, GC, ka, gp, SG, sd)
                allrows.append(features_along(mdl, lc, eng, w - WC, (GC, ka, gp),
                                              m0_product(eng, 1.0)))
            X = np.vstack(allrows)
            tab = contributions(lc, X, names)
            tables[(rtag, mdl)] = tab
            top = sorted(tab.items(), key=lambda kv: -kv[1][0])[:10]
            fam_tot = {}
            for term, (med, p95, cons) in tab.items():
                fam_tot[family(term)] = fam_tot.get(family(term), 0.0) + med
            print(f"[{rtag}] {mdl}: {len(tab)} active terms; family median-contribution totals: "
                  + ", ".join(f"{f}={v:.4f}" for f, v in sorted(fam_tot.items(), key=lambda kv: -kv[1])))
            for term, (med, p95, cons) in top:
                print(f"    {term:<28s} med={med:.4f} p95={p95:.4f} sign-cons={cons:.2f} [{family(term)}]")
            print(flush=True)
    np.savez_compressed("results/interp_contrib.npz", tables=np.array(tables, dtype=object))
    print("saved results/interp_contrib.npz")
