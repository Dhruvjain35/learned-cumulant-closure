# External reproduction protocol: frozen V7 on fresh disorder draws

Registered 2026-10-07, before any outside run and before the authors have simulated any panel cell with its
reserved seed. The package is `portable_v7/` (export commit `deee321`; the protocol commit is the one adding
this file).

## Objective

Can a researcher outside this project, in a clean environment, reproduce the frozen V7 closure's burst-peak
predictions and its advantage over second order? The run uses:
- the portable frozen model files (no scikit-learn, no pickle);
- disorder draws that no one has run before;
- the repository's stated model;
- no retraining, retuning or rerouting.

## Evidence levels (what each would license)

| Level | What the collaborator does | Wording it licenses |
|---|---|---|
| 1 | runs `run_selfcheck.py` and the panel with `run_external_v7_reproduction.py` | "an external collaborator executed the frozen V7 package in a clean environment and reproduced its predictions" |
| 2 (target) | Level 1, **plus** computes the exact reference for the same cells and draws with `julia/exact_panel.jl` (QuantumOptics.jl, a separate code path) | "…and its advantage over second order on fresh disorder draws, against an exact reference computed outside the project's Python stack" |
| 3 (not requested) | reimplements the closure, features and engine from the documentation | needed before "independent implementation" or "independent replication" can be used |

Neither Level 1 nor Level 2 is an independent implementation of V7. The learned model and the moment engine
are this project's code.

## Panel (deterministic; `portable_v7/external_panel.json`)

Selection rule, fixed before any panel run: for each (support class, routed branch) in {B, C, D} × {V4, V6},
take the Result 40 parameter cell with the lowest ID. Seeds 400–405 are assigned in that order.

| Panel id | Parameters (g_c, κ, γφ, σ) | Result 40 cell | Class | Branch | Reserved seed |
|---|---|---|---|---|---|
| X1 | 0.55, 0.10, 0.06, 0.10 | H01 | B, interpolation | V4 | 400 |
| X2 | 0.70, 0.035, 0.042, 0.15 | H09 | B, interpolation | V6 | 401 |
| X3 | 0.70, 0.36, 0.05, 0.15 | H14 | C, κ above range | V4 | 402 |
| X4 | 0.70, 0.025, 0.05, 0.15 | H13 | C, κ below range | V6 | 403 |
| X5 | 1.10, 0.20, 0.12, 0.15 | H25 | D, g_c and γφ above range | V4 | 404 |
| X6 | 0.70, 0.025, 0.015, 0.15 | H27 | D, κ and γφ below range | V6 | 405 |

- M = 5, γ↓ = 0.02, ω_c = 1, fully inverted initial state.
- X6's parameter point was Result 40's internal worst case (−1.46%). The ID rule selected it, not its
  performance; it is kept as a conservative challenge.
- The parameter points are known from Result 40, but every (cell, seed) pair here is new.

## Contamination statement

Seeds 400–405 appear nowhere else in the repository. The authors have **not** run, drawn or inspected any panel
cell with its reserved seed. The collaborator's run generates the draws from the seeds
(`numpy.random.default_rng(seed).normal(0.0, σ, 5)`, as in `pv7/engine/exact_tc.py`) and records every array.
Cross-platform RNG agreement is checked first, by the self-check's bit-for-bit draw for a published seed.

## Steps for the collaborator

1. Environment: Python 3.12; `pip install -r portable_v7/requirements-external.txt` (NumPy, SciPy, QuTiP only).
2. `cd portable_v7 && python run_selfcheck.py`. This is a **SELF-CHECK FIXTURE, NOT AN INDEPENDENT RESULT**
   (M = 2, published draw 50). It must print `SELF-CHECK PASS`. If it does not, stop and send
   `selfcheck_output/selfcheck.json`.
3. `python run_external_v7_reproduction.py`. It reruns the self-check, then the six cells: exact (QuTiP),
   plain second order, frozen V7. Expect a few minutes per cell at M = 5. Output goes to `external_output/`.
4. (Level 2) From the repository root, using its Julia environment:
   `julia --project=. portable_v7/julia/exact_panel.jl portable_v7/external_output/panel_params_for_julia.json`,
   then `python portable_v7/compare_julia_exact.py portable_v7/external_output`.
5. Return `external_output/` unedited (results.json, results.csv, run.log, environment.json,
   panel_params_for_julia.json and, for Level 2, julia_exact.csv and julia_comparison.json), plus a short note:
   name, date, machine/OS, and whether anything had to be changed. Any change must be described.

## No-adjustment rule

Do not retrain; move the router threshold; change clipping, features or model files; drop or add cells; change
seeds; or rerun a cell to get a different number. A technical failure is reported as such, together with any
installation fix needed.

## Reproduction criterion (fixed now)

1. **Software agreement:** `run_selfcheck.py` passes on the collaborator's machine. Its tolerances allow for
   cross-machine floating-point differences: draws exact, network outputs ≤ 1e-9, exact peak ≤ 1e-5 relative,
   closure peaks ≤ 1e-4 relative.
2. **Accuracy pattern on fresh draws:**
   - **REPRODUCED:** V7 is closer to exact than plain in 6/6 cells, median |err_V7| ≤ 3%, and V7 is stable in
     every cell.
   - **PARTIALLY REPRODUCED:** not REPRODUCED, but V7 is closer in ≥ 4/6 cells and median |err_V7| < 0.5 ×
     median |err_plain|.
   - **NOT REPRODUCED:** otherwise.
3. **Level 2 reference check:** the Julia and Python exact peaks agree to ≤ 1e-4 (relative) in every cell. If
   not, both closures' errors are reported against both references and the discrepancy is investigated before
   any wording is used.
4. Every number is archived as returned, whatever the outcome, under `external_validation/<date>-<name>/`, with
   a receipt-side audit like `external_validation/PROVENANCE.md`.

These bands are set from what is already known: Result 40's V7 errors are 0.1–1.5% and second order's are
28–33% on these parameter points. They are not chosen so that any likely outcome passes. For example, a V7
error near 5% in two cells would give PARTIALLY REPRODUCED.

## What a successful run would and would not license

- **Would (Level 2, REPRODUCED):** "An external collaborator ran the portable frozen V7 package in a clean
  environment on six held-out parameter points with fresh disorder draws, and, against an exact reference
  computed in QuantumOptics.jl, reproduced its burst-peak accuracy and its advantage over second order."
- **Would not:**
  - an independent implementation of V7;
  - experimental validation;
  - validation at any M other than 5 (size transfer is Results 34–38);
  - any initial state other than full inversion, any disorder family other than Gaussian, or γ↓ ≠ 0.02;
  - an "independently replicated" claim for anything.
- `EXTERNAL_VALIDATION_STATUS.md` changes only after the run returns and the receipt-side audit is committed.
