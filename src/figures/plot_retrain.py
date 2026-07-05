"""
Figure for the retrain result: closure trained on M<=5 running a system with M=8 and M=10 molecules it never
saw, on top of the exact (quantum-trajectory) reference, vs plain cumulant and the old M=2,3 closure.

Also pickles both fitted closures (results/lc_old.pkl, results/lc_new.pkl) so no future step ever refits.
Fits are deterministic (random_state=0, fixed caches), so a pickle load reproduces the eval exactly.
"""
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from local_closure import LocalClosure
from local_run import run, ClosureApply, observables, build
from mc_ref import mc_reference

WC, GCOLL = 1.0, 0.8
NC = 12
TEST_SEED, TEST_SIGMA = 7, 0.15
NTRAJ = 200


def get_closure(cache, pkl):
    if os.path.exists(pkl):
        with open(pkl, "rb") as f:
            return pickle.load(f)
    lc = LocalClosure(M=3)
    lc.fit(np.load(cache, allow_pickle=True)["data"].item())
    with open(pkl, "wb") as f:
        pickle.dump(lc, f)
    return lc


if __name__ == "__main__":
    lc_old = get_closure("results/transfer_train.npz", "results/lc_old.pkl")
    lc_new = get_closure("results/transfer_train_v2.npz", "results/lc_new.pkl")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for col, M in enumerate([8, 10]):
        eng, w, g = build(M, TEST_SIGMA, TEST_SEED)
        det = w - WC
        ts, tp = run(eng, None)
        _, to = run(eng, ClosureApply(lc_old, eng, det))
        _, tn = run(eng, ClosureApply(lc_new, eng, det))
        _, nref, zref, mcerr = mc_reference(M, NC, w, g, ntraj=NTRAJ)   # cached
        for rowi, (obs_i, lab) in enumerate([(0, "photon number"), (1, "mean inversion")]):
            ax = axes[rowi][col]
            for traj, c, ls, name in [(tp, "tab:red", "-", "plain cumulant"),
                                      (to, "tab:orange", "--", "closure (trained M=2,3)"),
                                      (tn, "tab:blue", "-", "closure (trained M=2-5)")]:
                n, z = observables(eng, traj)
                y = n if obs_i == 0 else z
                ax.plot(ts[:len(y)], y, c, ls=ls, lw=1.4, label=name)
            yr = nref if obs_i == 0 else zref
            ax.plot(ts[:len(yr)], yr, "k", lw=2.2, alpha=0.65, label="exact (mcsolve)", zorder=0)
            ax.set_ylabel(lab)
            if rowi == 0:
                ax.set_title(f"M = {M} molecules (never in training), disordered")
            if rowi == 1:
                ax.set_xlabel("time")
        print(f"M={M} plotted (MC err {mcerr:.3f})")
    axes[0][0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Learned cumulant closure at large M via the scalable engine (held-out disorder)")
    fig.tight_layout()
    fig.savefig("results/retrain_closure.png", dpi=150)
    print("saved results/retrain_closure.png")
