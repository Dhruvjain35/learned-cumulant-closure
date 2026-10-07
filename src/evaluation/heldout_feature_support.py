"""
POST-HOC companion to heldout_parameter_burst.py (NOT pre-registered; decides nothing in the P3 verdict).

P3's support classes are defined in raw physical parameters, and the pre-registration noted that the closure's
features are ratios to kappa (Delta/kappa, g_c/kappa, gamma_phi/kappa) with kappa raw. This script reports, per
evaluation cell, which of g_c/kappa, kappa, gamma_phi/kappa and sigma/kappa (a scale proxy for the per-site
Delta/kappa) lie outside the per-axis range spanned by the 81 training tuples, alongside the cell's V7 error.

    python run.py src/evaluation/heldout_feature_support.py
"""
import json
import numpy as np

from training_support import TRAIN

NAMES = ("g_c/kappa", "kappa", "gamma_phi/kappa", "sigma/kappa")


def ratios(t):
    g, k, gp, sg = t
    return np.array([g / k, k, gp / k, sg / k])


def main():
    F = np.array([ratios(t) for t in TRAIN])
    lo, hi = F.min(0), F.max(0)
    spec = json.load(open("results/heldout_param_cells.json"))
    rows = np.load("results/heldout_param_burst.npz", allow_pickle=True)["rows"]
    err = {r["id"]: r["v7"]["signed"] for r in rows}
    L = ["HELD-OUT CELLS IN RATIO-FEATURE COORDINATES (post-hoc, not pre-registered)",
         "training per-axis ranges: " + ", ".join(f"{n} [{a:.3g}, {b:.3g}]" for n, a, b in zip(NAMES, lo, hi)), ""]
    n_out = 0
    for c in spec["cells"]:
        v = ratios((c["g_c"], c["kappa"], c["gamma_phi"], c["sigma"]))
        out = [n for n, x, a, b in zip(NAMES, v, lo, hi) if x < a - 1e-12 or x > b + 1e-12]
        n_out += bool(out)
        L.append(f"{c['id']} class {c['cls']}  ratio-space outside range: {', '.join(out) if out else '-':34s} "
                 f"V7 signed error {err[c['id']]:+.4f}")
    L.append(f"\n{n_out} of {len(spec['cells'])} cells lie outside the training range on at least one ratio axis.")
    print("\n".join(L))
    with open("results/heldout_feature_support.log", "w") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
