<div align="center">

# Learned Cumulant Closure

### Recovering the superradiant burst where the standard cumulant hierarchy breaks

Second-order cumulants **overshoot** the superradiant burst peak. Fourth-order **runs away**. A small,
structure-preserving **learned third-cumulant closure** stays stable and tracks the exact reference — in the
open, *disordered* cavity–emitter regime where the standard truncation hierarchy doesn't. Every extension that
*didn't* survive is kept in the record.

![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)
&nbsp;![Julia](https://img.shields.io/badge/julia-≥1.9-9558B2?logo=julia&logoColor=white)
&nbsp;![Reference](https://img.shields.io/badge/reference-exact_QuTiP-1f77b4)
&nbsp;![Science](https://img.shields.io/badge/science-frozen-2ca02c)
&nbsp;![License](https://img.shields.io/badge/license-MIT-blue)

</div>

---

## TL;DR

> On the **superradiant burst peak** of an open, disordered Tavis–Cummings ensemble — a measurable
> collective-emission observable — the standard second-order cumulant closure is off by ≈ **+29%**, the tested
> κ4 = 0 third-order closure by ≈ **−28%** (and it runs away in the low-loss pocket), and the κ5 = 0 fourth-order
> closure **runs away** — its integration triggers a divergence guard. A frozen, physics-routed **learned closure
> (V7)** stays within ≈ **±1%** of the
> exact reference in the tested cells. This repo has the engines, the learned closure, the mechanism analysis,
> the rejected hypotheses, and a cross-platform reproduction package.

## The result in one table

Signed burst-peak error vs. an exact master-equation solve, in the evaluated disordered open-cavity cells:

| closure | burst peak vs. exact | stability | cost |
|---|---|---|---|
| mean-field | no burst from these ICs | — | trivial |
| **2nd-order cumulant** | overshoots ≈ **+29%** | stable | no high-order engine to construct |
| κ4 = 0 (3rd-order) | undershoots ≈ **−28%** | runaway growth (guard triggered) in the low-loss pocket | heavier than 2nd; M = 4 pocket case failed |
| κ5 = 0 (4th-order) | overshoots, then **runaway growth** | guard triggered in all M = 3 cells (both regimes) | substantial, highly variable at M = 3 |
| **V7 (learned)** | ≈ **±1%** | stable | no high-order engine to construct; per-rollout time not separately logged |

Among the tested zero-next-cumulant truncations through fourth order, none produced a stable, accurate burst
closure in these cells. *(Scope: exact-referenced at the tested molecule counts; behaviour beyond the exact
ceiling is reported as an estimate, not a result.)*

## V7 against the closure ladder

<div align="center">
<img src="assets/img/ladder.png" width="90%" alt="Closure-structure ladder: stability and median NRMSE for mean-field, plain 2nd-order, linear, sparse-quadratic, kappa4=0 third-order, and V7 against the exact reference, across regimes."/>
</div>

*Stability (%) and median conditional NRMSE for each closure across regimes (in-box / seam / sanity / far-M),
measured against the exact reference on held-out disorder draws. V7 (right column) carries the lowest error while
staying stable everywhere it is defined.*

## Why it fails, mechanistically

The burst-peak error is carried by **two connected third-cumulant channels**: photon-number–inversion
`⟨a†a σᶻ⟩` (dominant) and photon-polarization–inversion `⟨a σ⁺ σᶻ⟩` (secondary). Second order sets both to
zero (→ overshoot); κ4 = 0 mis-signs them (→ undershoot); V7 reconstructs them.

<div align="center">
<img src="assets/img/mechanism.png" width="92%" alt="Mechanism figure: which kappa3 channels carry the correction, the dominant channel's kappa3 trajectory for exact vs V7 vs T3 vs plain, and closed-loop ablation."/>
</div>

*(a) The two channels that carry the correction. (b) The dominant channel's connected κ3 through the burst —
exact and V7 overlie; κ4 = 0 (T3) mis-estimates it; plain sits at zero. (c) Closed-loop ablation: removing
either channel degrades or destabilizes V7. See `docs/FINDINGS.md`, Result 25.*

## Runtime — is V7 actually faster?

A frozen wall-clock benchmark on the same M=3 cells (build vs. rollout vs. total; medians over the evaluated
cells, one machine — full method and caveats in
[`results/runtime_baseline_comparison.md`](results/runtime_baseline_comparison.md)):

<div align="center">
<img src="results/runtime_benchmark_fig.png" width="82%" alt="Frozen-runtime benchmark: build and rollout wall-clock (log scale) for 2nd-order, V7, kappa4=0 (T3), and kappa5=0 (T4) on M=3 cells; V7 build matches 2nd-order, V7 total sits far below the higher-order truncations."/>
</div>

V7 reuses the second-order engine's ≈ 3 s build and adds a learned per-step evaluation instead of constructing a
higher-order engine, so its median **total** (≈ 4.6 s) sits close to second order (≈ 3.1 s) and roughly **4×**
and **18×** below the κ4 = 0 (≈ 21 s) and κ5 = 0 (≈ 81 s) truncations. Its **per-rollout** cost (≈ 1.5 s) is ≈ 76×
the second-order rollout — measured at M = 3 on one machine, so this is a build-dominated total-time advantage,
**not** a "cheap," "fast," or "near-second-order" claim.

## Repository layout

```
src/
├── engines/       exact Tavis–Cummings (QuTiP), 2nd-order cumulant, scalable EOM, mcsolve reference
├── closures/      the learned closures (V4 memoryless, V6 memory-augmented) + V7 router
├── baselines/     kappa4=0 (3rd) and kappa5=0 (4th) truncations, linear/quadratic baselines
├── data_gen/      training-data generation from exact moment trajectories
├── training/      closure fitting + retrain-one-rung-up (M-ladder bootstrap)
├── evaluation/    pilot / campaign / burst-peak / transfer evaluation harnesses
├── analysis/      mechanism attribution + ablation, disorder sweep, interpretability, criterion tests
├── validation/    engine gates and equivalence checks
├── figures/       figure-generating scripts
└── tools/         export exact replication parameters
scripts/           runtime benchmark, frozen-trajectory export, and plotting utilities (read-only over results)
docs/              FINDINGS (every result incl. rejected hypotheses), PROTOCOL, REPLICATION package
results/           figures, small result artifacts, frozen V7 model pickles, replication params
replicate_cumulant.jl · julia_setup.jl   cross-platform reproduction (Julia)
run.py · Makefile                        run any src/ module from the repo root
```

## Run it

**Reproduce the baseline in a separate codebase (Julia — the collaborator path):**

```bash
julia --project=. julia_setup.jl         # one-time install
julia --project=. replicate_cumulant.jl  # exact vs 2nd-order burst peak → results/replication_cumulant.csv
```

Start with **[`docs/REPLICATION_QUICKSTART.md`](docs/REPLICATION_QUICKSTART.md)**; record results in
**[`docs/REPLICATION_SIGNOFF.md`](docs/REPLICATION_SIGNOFF.md)**; assumptions and mismatch risks are in
**[`docs/REPLICATION_AUDIT.md`](docs/REPLICATION_AUDIT.md)**. The Julia script has **not** been executed by the
author and is not asserted to be validated — that judgement is the collaborator's to record.

**Python side** (run from the repo root; `run.py` puts the flat imports on the path):

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
make test               # import smoke test
make burst              # exact vs 2nd-order vs V7 burst peak
make mechanism          # connected-kappa3 channel attribution + ablation
# or run any module directly:
python run.py src/baselines/t4_eval.py
```

## Scope and discipline

- **Exact-referenced only where exact references exist.** Rollouts beyond that range are numerical reach, not
  demonstrated accuracy, and are labelled as such throughout.
- **Frozen after evaluation.** V7 and its two branches were frozen before final testing; pre-registrations and
  held-out disorder draws are in [`docs/PROTOCOL.md`](docs/PROTOCOL.md).
- **Rejected hypotheses are kept.** A once-calibrated parameter-only routing criterion, an M-scaling story, and a
  disorder-dependence story were each tested and did **not** survive — they live in
  [`docs/FINDINGS.md`](docs/FINDINGS.md) (Results 22–24, 26) as documented negatives, e.g.:

  <div align="center">
  <img src="assets/img/disorder_null.png" width="80%" alt="Disorder sweep: burst observables and connected-kappa3 channels vs disorder, with error bars; the effect is monotone, not the hypothesized non-monotonic structure."/>
  </div>

  *A disorder-dependence hypothesis, tested and rejected: everything resolved moves monotonically with disorder,
  with no non-monotonic structure — reported as the null it is.*

- **No claim of exclusivity.** Tensor-network, trajectory, and higher-order-closure methods can address parts of
  this problem; this repository reports what these specific closures do in these specific cells.

- The physics model, the validation targets, and every reported number come from the runnable code in this repository, gated by the checks in
  [`docs/PROTOCOL.md`](docs/PROTOCOL.md).

## License

MIT — see [`LICENSE`](LICENSE).
