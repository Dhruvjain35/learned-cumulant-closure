# portable_v7: the frozen V7 closure without sklearn or pickle

This directory is a self-contained, non-executable export of the **frozen** V7 learned third-cumulant closure,
plus a small NumPy/SciPy/QuTiP package (`pv7/`) that runs it. Nothing was retrained. Every array is copied out of
the frozen estimators (`results/lc_v3dens.pkl`, sha256 `be2dde48…98f81`; `results/lc_v6.pkl`, sha256
`0599643e…b800b86`). The equivalence gates show the portable path reproduces the originals bit for bit on the
same machine.

| Path | What it is |
|---|---|
| `models/v4/networks.npz`, `models/v6/networks.npz` | per canonical cumulant type: MLP weights `W0..W2`, biases `b0..b2`, scaler `mean`/`scale`, output clip bounds `lo`/`hi` (float64; read with `allow_pickle=False`) |
| `models/*/metadata.json` | architecture (input → 128 → 128 → 2, ReLU, identity output), type list, source-pickle hash |
| `feature_spec.json` | the 21 (V4) / 27 (V6) inputs in column order, the ratio transform, canonical factor ordering, conjugation and self-conjugate rules, clipping, the V6 memory dynamics |
| `router.json` | V6 iff κ ≤ 0.045 **and** γφ ≤ 0.05 (inclusive), else V4; decided once at t = 0 |
| `pv7/` | portable inference: `features.py`, `model.py`, `closure.py`, `router.py`, `rollout.py`; `engine/` is the moment engine vendored verbatim from the source repository, with only its imports made package-relative |
| `metadata.json` | provenance: source commit, pickle and source-file hashes, export environment |
| `SHA256SUMS` | hashes of every file above |

Requirements: Python 3.12, NumPy, SciPy and QuTiP. No scikit-learn, no joblib, and no pickle loading.

Equivalence with the frozen originals is checked in the source repository by `make portable-test`
(`tests/test_portable_v7.py`). It covers vendored-engine identity, router boundaries, the feature spec, network
arrays and clip bounds, features against the training-time builders, κ3 on direct, conjugated and
self-conjugate keys, static predictions, and full V4- and V6-routed rollouts.

External-run instructions are in `docs/EXTERNAL_V7_REPRODUCTION_PROTOCOL.md`.
