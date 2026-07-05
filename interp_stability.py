"""
Interpretability step 4 (PROTOCOL): grouped stability selection. Refit the sparse quadratic repeatedly on
TRAINING data resampled by TRAJECTORY FAMILY (config point-ids with replacement -- never row-level
bootstrap, which would leak trajectory structure across resamples), and record per-term selection
frequency and sign consistency. Terms that appear in one fit are not physics; terms that survive
resampling and matter in rollout are candidates. Conjugate-paired cumulant types are handled as one
group by construction (only canonical types are trained). Dominant types: (sp,sm,sz) and (a,sp,sz).
"""
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline

from closure_v3 import ratio_transform
from interp_contrib import NAMES21, NAMES27

POOL = "results/train_v6.npz"
TYPES = [("sp", "sm", "sz"), ("a", "sp", "sz")]
B = 15
ROW_CAP = 150_000
RNG = np.random.default_rng(41)

if __name__ == "__main__":
    pool = np.load(POOL, allow_pickle=True)
    data = pool["data"].item()
    results = {}
    for typ in TYPES:
        X, Y, P = data[typ]
        pids = np.unique(P)
        for cols, names, tag in ((21, NAMES21, "q21"), (27, NAMES27, "q27")):
            sel = {}
            signs = {}
            for b in range(B):
                boot = RNG.choice(pids, len(pids), replace=True)
                mask = np.isin(P, boot)
                idx = np.nonzero(mask)[0]
                if len(idx) > ROW_CAP:
                    idx = RNG.choice(idx, ROW_CAP, replace=False)
                Xi = ratio_transform(X[idx][:, :cols])
                pipe = Pipeline([("sc", StandardScaler()),
                                 ("poly", PolynomialFeatures(2, include_bias=False)),
                                 ("lasso", Lasso(alpha=1e-4, max_iter=5000))])
                pipe.fit(Xi, Y[idx])
                W = pipe.named_steps["lasso"].coef_
                tn = pipe.named_steps["poly"].get_feature_names_out(names)
                for j in range(W.shape[1]):
                    w = W[:, j]
                    if np.any(np.abs(w) > 1e-10):
                        sel[tn[j]] = sel.get(tn[j], 0) + 1
                        signs.setdefault(tn[j], []).append(np.sign(w[np.argmax(np.abs(w))]))
                print(f"  {typ} {tag} resample {b+1}/{B}: {int(np.sum(np.any(np.abs(W)>1e-10, axis=0)))} "
                      f"active terms", flush=True)
            stable = {t: (c / B, float(np.mean(np.array(signs[t]) == np.sign(np.median(signs[t])))))
                      for t, c in sel.items()}
            results[(typ, tag)] = stable
            top = sorted(stable.items(), key=lambda kv: (-kv[1][0], -kv[1][1]))[:15]
            print(f"\n{typ} {tag}: terms with selection freq >= 0.8 and sign-cons >= 0.9: "
                  f"{sum(1 for _, (f, s) in stable.items() if f >= 0.8 and s >= 0.9)}")
            for t, (f, s) in top:
                print(f"    {t:<28s} freq={f:.2f} sign-cons={s:.2f}")
            print(flush=True)
    np.savez_compressed("results/interp_stability.npz", results=np.array(results, dtype=object))
    print("saved results/interp_stability.npz")
