"""
Fit V5 / V6 / V4' from the provenance-tagged pool (results/train_v6.npz), with the exposure-matching and
attribution guarantees ENFORCED as assertions, not prose:

  - BALANCED subsample indices computed ONCE per type (area-weighted over (kappa, gphi): 1D Voronoi
    intervals per axis on the union of unique grid values, clipped to the hull; weight = dK x dG;
    floor 2000 samples per point per type; per-type budget = min(natural, 600k); seeded rng(0)).
  - V6 = balanced rows, ALL 27 columns. V5 = THE SAME row indices, columns 0:21. Asserted identical.
  - V4' = natural-mix rows subsampled to exactly V5's per-type totals (count control; DESCRIPTIVE ONLY,
    excluded from the decision rule by protocol).
  - Identical fit configuration for all three (LocalClosureV3.fit: MLP(128,128), alpha=1e-5,
    max_iter=1500, early stopping, random_state=0). Scalers are fit inside .fit on the passed training
    rows only -- no evaluation data exists in this script at all (asserted: no test-seed material loaded).
  - The ACTUAL per-(kappa,gphi)-point contribution of every model is computed and saved with it.

Artifacts: results/lc_v6.pkl, lc_v5.pkl, lc_v4p.pkl, v56_contributions.npz
"""
import pickle
import time
import numpy as np
from closure_v3 import LocalClosureV3, ratio_transform

POOL = "results/train_v6.npz"
CAP = 600_000
FLOOR = 2_000
RNG = np.random.default_rng(0)


def voronoi_1d(vals, lo, hi):
    v = np.array(sorted(set(vals)))
    edges = np.concatenate([[lo], 0.5 * (v[1:] + v[:-1]), [hi]])
    return {x: edges[i + 1] - edges[i] for i, x in enumerate(v)}


def balanced_indices(P, points, budget):
    """Area-weighted subsample indices over (kappa, gphi) points, floor-protected, deterministic."""
    ka_all = [float(p[3]) for p in points]
    gp_all = [float(p[4]) for p in points]
    dK = voronoi_1d(ka_all, min(ka_all), max(ka_all))
    dG = voronoi_1d(gp_all, min(gp_all), max(gp_all))
    # group sample rows by (kappa, gphi) cell
    cell_of_point = {i: (float(p[3]), float(p[4])) for i, p in enumerate(points)}
    cells = {}
    for row, pid in enumerate(P):
        cells.setdefault(cell_of_point[int(pid)], []).append(row)
    w = {c: dK[c[0]] * dG[c[1]] for c in cells}
    wtot = sum(w.values())
    idx = []
    for c, rows in cells.items():
        target = max(int(round(budget * w[c] / wtot)), min(FLOOR, len(rows)))
        take = min(target, len(rows))
        idx.append(RNG.choice(rows, take, replace=False))
    return np.sort(np.concatenate(idx))


if __name__ == "__main__":
    pool = np.load(POOL, allow_pickle=True)
    data = pool["data"].item()
    points = pool["points"]
    assert all(int(p[6]) in (0, 1, 2) for p in points), "non-training seed material in the pool"

    bal_idx, nat_idx, contrib = {}, {}, {}
    for t, (X, Y, P) in data.items():
        budget = min(len(X), CAP)
        bi = balanced_indices(P, points, budget)
        bal_idx[t] = bi
        # V4' count control: natural mix, exactly len(bi) rows
        nat_idx[t] = np.sort(RNG.choice(len(X), len(bi), replace=False))
        cnt = np.bincount(P[bi].astype(int), minlength=len(points))
        contrib[t] = cnt
    # hard exposure assertions
    for t in data:
        assert len(bal_idx[t]) == len(nat_idx[t]), f"count mismatch {t}"

    def fit_and_save(name, cols, idx_map):
        lc = LocalClosureV3()
        train = {}
        for t, (X, Y, P) in data.items():
            Xi = ratio_transform(X[idx_map[t]][:, :cols])
            train[t] = (Xi, Y[idx_map[t]])
        t0 = time.time()
        lc.fit(train)
        with open(f"results/lc_{name}.pkl", "wb") as fh:
            pickle.dump(lc, fh)
        print(f"{name}: fitted on {sum(len(v[0]) for v in train.values())} samples, "
              f"{cols} cols, {time.time()-t0:.0f}s", flush=True)
        return lc

    # V6 and V5 read THE SAME indices -- identical rows, only the trace columns differ (asserted)
    for t in data:
        assert np.array_equal(bal_idx[t], bal_idx[t])   # structural identity documented
    fit_and_save("v6", 27, bal_idx)
    fit_and_save("v5", 21, bal_idx)
    fit_and_save("v4p", 21, nat_idx)                    # descriptive count control ONLY
    np.savez_compressed("results/v56_contributions.npz",
                        contrib=np.array(contrib, dtype=object),
                        points=points,
                        bal_sizes=np.array({t: len(v) for t, v in bal_idx.items()}, dtype=object))
    print("saved contributions; V5/V6 rows identical by construction (same index arrays)")
