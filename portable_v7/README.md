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
| `external_panel.json` | the six-cell external panel (reserved seeds 400–405) and its selection rule |
| `run_selfcheck.py`, `selfcheck_expected.json` | SELF-CHECK FIXTURE, NOT AN INDEPENDENT RESULT: M = 2, published draw 50, with expected values produced by the original pickle stack |
| `run_external_v7_reproduction.py` | the panel runner: exact (QuTiP), plain second order and V7 → `external_output/` |
| `julia/exact_panel.jl`, `compare_julia_exact.py` | optional Level 2: exact reference in QuantumOptics.jl for the same cells and draws |
| `requirements-external.txt` | the only Python dependencies (NumPy, SciPy, QuTiP) |
| `SHA256SUMS` | hashes of every file above |

Requirements: Python 3.12, NumPy, SciPy and QuTiP. No scikit-learn, no joblib, and no pickle loading.

Equivalence with the frozen originals is checked in the source repository by `make portable-test`
(`tests/test_portable_v7.py`). It covers vendored-engine identity, router boundaries, the feature spec, network
arrays and clip bounds, features against the training-time builders, κ3 on direct, conjugated and
self-conjugate keys, static predictions, and full V4- and V6-routed rollouts.

External-run instructions are in `docs/EXTERNAL_V7_REPRODUCTION_PROTOCOL.md`.

## Running it (external collaborator)

```bash
python3.12 -m venv venv && . venv/bin/activate
pip install -r requirements-external.txt          # NumPy, SciPy, QuTiP only
python run_selfcheck.py                           # SELF-CHECK FIXTURE -- NOT AN INDEPENDENT RESULT; must PASS
python run_external_v7_reproduction.py            # the six-cell panel -> external_output/
# optional (Level 2), from the repository root:
julia --project=. portable_v7/julia/exact_panel.jl portable_v7/external_output/panel_params_for_julia.json
python portable_v7/compare_julia_exact.py portable_v7/external_output
```

QuTiP prints "matplotlib not found: Graphics will not work". That is harmless; nothing here plots. Run the
scripts from this directory; no `PYTHONPATH` setting is needed. The full protocol, the panel, and the
pre-declared reproduction criterion are in `docs/EXTERNAL_V7_REPRODUCTION_PROTOCOL.md`.
