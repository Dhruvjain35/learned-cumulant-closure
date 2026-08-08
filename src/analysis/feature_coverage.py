"""
Why transfer works, why it is bounded, and why retraining one rung up repairs it.

transfer_collapse.py established that kappa3 is very nearly a single function of the estimator's
own local features -- binned in (<a^d a>, <sz_i>) the across-M spread is only ~1.1x the within-M
scatter -- while in the intensive context variables (n/M, zbar) it is not (~3x). So the estimator
is not size-independent because its inputs are all intensive. It is size-independent because it
conditions directly on the raw pair moment <a^d a>, and given that moment the residual dependence
on M is weak.

That has a sharp consequence, which this script measures.

<a^d a> is EXTENSIVE: the burst peak grows roughly linearly in M. So training at small M covers a
small window of <a^d a>, and running at larger M pushes the model into feature values it never
saw. Transfer should then hold while the test features stay inside the training window and decay
once they leave it -- not because the physics changes, but because the model is extrapolating in
its own inputs.

This predicts the ladder behaviour already documented in FINDINGS: strong transfer from M=2,3 to
M=4,5, decay toward parity by M=10, and recovery when the model is retrained on M<=5. Here we test
the prediction directly by measuring what fraction of each target M's samples lie inside the
feature range spanned by a given training set.

Run:  python src/analysis/feature_coverage.py
"""
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from kappa3_trace import solve_cell, NCS  # noqa: E402
from transfer_collapse import features_and_target  # noqa: E402


def coverage(train, test, keys=("npair", "zi"), pad=0.0):
    """Fraction of `test` samples inside the axis-aligned range spanned by `train`."""
    inside = np.ones(test[keys[0]].size, bool)
    for k in keys:
        lo, hi = train[k].min(), train[k].max()
        span = hi - lo
        inside &= (test[k] >= lo - pad * span) & (test[k] <= hi + pad * span)
    return float(inside.mean())


def main():
    Ms = [2, 3, 4, 5]
    data = {}
    print("Feature-range coverage: how much of a larger system is INTERPOLATION for a")
    print("model trained on smaller ones, in the estimator's own coordinates?")
    print()
    for M in Ms:
        S = solve_cell(M, NCS[M], tmax=40.0, nt=401)
        data[M] = features_and_target(S)
        d = data[M]
        print(f"  M={M}: <a^d a> spans [{d['npair'].min():.3f}, {d['npair'].max():.3f}], "
              f"|kappa3| max {np.abs(d['k3']).max():.4f}")

    def merge(ms):
        out = {}
        for k in data[ms[0]]:
            out[k] = np.concatenate([data[m][k] for m in ms])
        return out

    print()
    print("  training set        target M    inside training feature range")
    for train_ms in ([2, 3], [2, 3, 4, 5]):
        tr = merge(train_ms)
        for M in Ms:
            if M in train_ms:
                continue
            c = coverage(tr, data[M])
            print(f"  M={str(train_ms):<12}      {M}            {100 * c:6.1f}%")
        # extrapolate the trend: where would a larger M land?
    print()

    # How far outside does each M sit, measured in training-window widths?
    tr23 = merge([2, 3])
    lo, hi = tr23["npair"].min(), tr23["npair"].max()
    width = hi - lo
    print("  peak <a^d a> vs the M=2,3 training window "
          f"[{lo:.3f}, {hi:.3f}] (width {width:.3f}):")
    for M in Ms:
        pk = data[M]["npair"].max()
        over = (pk - hi) / width
        print(f"    M={M}: peak {pk:6.3f}   {'inside' if over <= 0 else f'{over:.2f} window-widths beyond'}")

    # A high overall coverage can be misleading: most of a trajectory sits at low photon number,
    # where every M overlaps. What matters is coverage in the burst-peak region, because the peak
    # is the observable being claimed. Restrict to samples in the top decile of <a^d a>.
    print()
    print("  coverage restricted to the burst-peak region (top decile of <a^d a>):")
    for M in (4, 5):
        d = data[M]
        thr = np.quantile(d["npair"], 0.9)
        sel = d["npair"] >= thr
        sub = {k: v[sel] for k, v in d.items()}
        c_all = coverage(tr23, d)
        c_pk = coverage(tr23, sub)
        print(f"    M={M}: whole trajectory {100 * c_all:5.1f}%   "
              f"burst-peak region {100 * c_pk:5.1f}%")
    print("    -> the extrapolation is concentrated exactly where the claim is made.")

    os.makedirs("results", exist_ok=True)
    np.savez_compressed(
        "results/feature_coverage.npz",
        Ms=np.array(Ms),
        peak_npair=np.array([data[M]["npair"].max() for M in Ms]),
        peak_k3=np.array([np.abs(data[M]["k3"]).max() for M in Ms]),
        cov_from_23=np.array([coverage(merge([2, 3]), data[M]) for M in Ms]),
    )
    print("\n  saved results/feature_coverage.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
