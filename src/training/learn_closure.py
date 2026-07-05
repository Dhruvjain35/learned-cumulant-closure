"""
Test whether the closure map (low-order moments -> 3rd cumulant) is LEARNABLE and TRANSFERABLE.

Three splits:
  (1) random           : is it a function at all?
  (2) M-extrapolation  : train on M in {3,4}, test on M=5  (the whole point: generalize in emitter number)
  (3) sigma-extrapolation: train on low disorder, test on high disorder

Baselines:
  zero    : predict kappa3 = 0  (exactly what 2nd-order cumulant assumes)
  linear  : Ridge regression (is nonlinearity needed?)
  mlp     : small permutation-equivariant MLP on the local feature vector

Metric: skill = 1 - MSE(model)/MSE(zero).  skill>0 means the model extracts real closure signal
beyond the cumulant truncation; skill->1 means it fully captures it. R2 also reported.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


def load():
    d = np.load("results/closure_data.npz")
    return d["X"], d["Y"], d["M"], d["sigma"]


def skill_vs_zero(y_true, y_pred):
    mse_model = np.mean((y_true - y_pred) ** 2)
    mse_zero = np.mean(y_true ** 2)   # predicting 0
    return 1.0 - mse_model / mse_zero if mse_zero > 0 else float("nan")


def evaluate(Xtr, Ytr, Xte, Yte, name):
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    out = {}
    # zero baseline
    out["zero"] = dict(skill=skill_vs_zero(Yte, np.zeros_like(Yte)), r2=r2_score(Yte, np.zeros_like(Yte)))
    # linear
    lin = Ridge(alpha=1.0).fit(Xtr_s, Ytr)
    yl = lin.predict(Xte_s)
    out["linear"] = dict(skill=skill_vs_zero(Yte, yl), r2=r2_score(Yte, yl))
    # mlp
    mlp = MLPRegressor(hidden_layer_sizes=(128, 128), activation="relu", alpha=1e-4,
                       max_iter=800, early_stopping=True, n_iter_no_change=20, random_state=0)
    mlp.fit(Xtr_s, Ytr)
    ym = mlp.predict(Xte_s)
    out["mlp"] = dict(skill=skill_vs_zero(Yte, ym), r2=r2_score(Yte, ym))
    print(f"\n[{name}]  train={len(Xtr)}  test={len(Xte)}   (target |kap3| std={Yte.std():.4f})")
    for k in ["zero", "linear", "mlp"]:
        print(f"   {k:7s}: skill_vs_zero={out[k]['skill']:+.3f}   R2={out[k]['r2']:+.3f}")
    return out, (Yte, ym)


def main():
    X, Y, M, sig = load()
    print(f"loaded X={X.shape} Y={Y.shape}  M in {sorted(set(M))}  sigma in {sorted(set(np.round(sig,3)))}")

    # (1) random split
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X)); cut = int(0.8 * len(X))
    tr, te = idx[:cut], idx[cut:]
    evaluate(X[tr], Y[tr], X[te], Y[te], "random 80/20")

    # (2) M-extrapolation: train M in {3,4}, test M=5
    tr = np.where((M == 3) | (M == 4))[0]; te = np.where(M == 5)[0]
    _, (yt, yp) = evaluate(X[tr], Y[tr], X[te], Y[te], "M-extrapolation: train M{3,4} -> test M5")

    # (3) sigma-extrapolation: train low disorder, test high
    tr = np.where(sig <= 0.10 + 1e-9)[0]; te = np.where(sig >= 0.20 - 1e-9)[0]
    evaluate(X[tr], Y[tr], X[te], Y[te], "sigma-extrapolation: train sigma<=0.10 -> test sigma>=0.20")

    # plot the hardest generalization (M-extrap) predicted vs true
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 4.3))
        for c in range(2):
            ax[c].scatter(yt[:, c], yp[:, c], s=3, alpha=0.25)
            lim = np.abs(yt[:, c]).max() * 1.1
            ax[c].plot([-lim, lim], [-lim, lim], "k--", lw=1)
            ax[c].set_xlabel("true kappa3"); ax[c].set_ylabel("predicted")
            ax[c].set_title(f"M-extrapolation (train 3,4 -> test 5), {'Re' if c==0 else 'Im'}")
            ax[c].grid(alpha=.3)
        fig.tight_layout(); fig.savefig("results/closure_learn_Mextrap.png", dpi=130)
        print("\nsaved results/closure_learn_Mextrap.png")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
