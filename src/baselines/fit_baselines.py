"""
Baseline fits (PROTOCOL.md Phase B): L21/L27 (Ridge) and Q21/Q27 (sparse quadratic) on THE SAME balanced
rows as V5/V6. Row identity is not asserted rhetorically: the balanced indices are reconstructed by
replaying fit_v56's exact deterministic sequence (module RNG(0), same per-type call order), and the
resulting per-point contributions must EQUAL the saved v56_contributions before any fit runs.

Q fits use a seeded 300k/type subsample of the same rows (Lasso on 2M x 250 poly features is not
tractable); alpha fixed a priori at 1e-4 (PROTOCOL). All runtimes and parameter counts recorded.
"""
import pickle
import time
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

from closure_v3 import ratio_transform
from baseline_shell import LocalClosureShell
import fit_v56
from fit_v56 import balanced_indices

POOL = "results/train_v6.npz"


def nparams(model):
    if isinstance(model, Pipeline):
        model = model.named_steps["lasso"]
    if hasattr(model, "coef_"):
        c = np.asarray(model.coef_)
        return int(c.size) + int(np.asarray(model.intercept_).size)
    return -1


if __name__ == "__main__":
    pool = np.load(POOL, allow_pickle=True)
    data = pool["data"].item()
    points = pool["points"]

    # ---- reproduce fit_v56's balanced indices EXACTLY (replay module RNG sequence) and verify ----
    fit_v56.RNG = np.random.default_rng(0)
    bal_idx, nat_idx = {}, {}
    for t, (X, Y, P) in data.items():                       # same dict order as fit_v56 (insertion order)
        budget = min(len(X), fit_v56.CAP)
        bal_idx[t] = balanced_indices(P, points, budget)
        nat_idx[t] = np.sort(fit_v56.RNG.choice(len(X), len(bal_idx[t]), replace=False))
    saved = np.load("results/v56_contributions.npz", allow_pickle=True)["contrib"].item()
    for t in data:
        cnt = np.bincount(data[t][2][bal_idx[t]].astype(int), minlength=len(points))
        assert np.array_equal(cnt, saved[t]), f"row identity FAILED for type {t} -- do not fit"
    print("row identity vs saved v56 contributions: VERIFIED (identical balanced rows)\n", flush=True)

    rng = np.random.default_rng(7)                          # Q-subsample seed, fixed a priori
    specs = {
        "l21": (21, lambda: Ridge(alpha=1e-6), None),
        "l27": (27, lambda: Ridge(alpha=1e-6), None),
        "q21": (21, lambda: Pipeline([("poly", PolynomialFeatures(2, include_bias=False)),
                                      ("lasso", Lasso(alpha=1e-4, max_iter=5000))]), 300_000),
        "q27": (27, lambda: Pipeline([("poly", PolynomialFeatures(2, include_bias=False)),
                                      ("lasso", Lasso(alpha=1e-4, max_iter=5000))]), 300_000),
    }
    meta = {}
    for name, (cols, make, sub) in specs.items():
        t0 = time.time()
        train = {}
        for t, (X, Y, P) in data.items():
            idx = bal_idx[t]
            if sub is not None and len(idx) > sub:
                idx = np.sort(rng.choice(idx, sub, replace=False))
            train[t] = (ratio_transform(X[idx][:, :cols]), Y[idx])
        lc = LocalClosureShell()
        lc.fit(train, make_net=make)
        dt = time.time() - t0
        npar = {t: nparams(lc.nets[t]) for t in lc.nets}
        with open(f"results/lc_{name}.pkl", "wb") as fh:
            pickle.dump(lc, fh)
        nz = ""
        if name.startswith("q"):
            nzc = {t: int(np.sum(np.abs(lc.nets[t].named_steps["lasso"].coef_) > 1e-10))
                   for t in lc.nets}
            nz = f"  nonzero coefs: {sum(nzc.values())} total"
        print(f"{name}: {cols} cols, fit {dt:.0f}s, params/type ~{int(np.mean(list(npar.values())))}{nz}",
              flush=True)
        meta[name] = dict(fit_s=dt, nparams=npar)
    np.savez_compressed("results/baseline_fit_meta.npz", meta=np.array(meta, dtype=object))
    print("\nsaved 4 baseline models + meta")
