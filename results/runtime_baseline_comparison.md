# Frozen-runtime benchmark — build vs rollout vs total wall-clock

Measures whether V7 has a real, measured runtime advantage over the tested conventional closures on the **same
M=3 cells** used in the fourth-order fairness test. No retrain, reroute, tune, or model change — timing only.

Data: [`runtime_baseline_comparison.csv`](runtime_baseline_comparison.csv) ·
figure: [`runtime_benchmark_fig.png`](runtime_benchmark_fig.png) ·
script: [`../scripts/benchmark_rollout_runtime.py`](../scripts/benchmark_rollout_runtime.py)

## Cells and disorder draws

- **Regimes:** CONTROL (g_c, κ, γφ) = (0.70, 0.15, 0.02); POCKET / long-coherence = (0.70, 0.03, 0.02). γ↓ = 0.02, σ = 0.15.
- **Molecule count:** M = 3.
- **Disorder draws:** 50, 51, 52 (deterministic), each regime → six cells total.
- **V7 routing:** CONTROL → V4 branch; POCKET → V6 branch (fixed by parameters; not changed here).

## Definitions

- **build time** — constructing everything needed *before* time-integration starts.
  - 2nd-order / V7: the scalable engine (`build_cell`); V7 additionally constructs the closure applier. The
    frozen model pickles are loaded once, outside the timed loop.
  - κ4=0 (T3) / κ5=0 (T4): the truncation engine (`T3Engine` / `T4Engine`).
- **rollout time** — the time-integration itself (`run_from` / `run_v6` / `solve_ivp`). For κ5=0 (T4) the rollout
  is timed until the runaway-growth guard triggers; the guard's simulation-time *t* is reported separately.
- **total** — build + rollout.

## Repeats and environment

- Stable methods (2nd-order, V7, κ4=0 T3): one untimed warm-up, then **5 timed repeats per cell**.
- κ5=0 (T4): **1 timed run per cell** (build 72–90 s here, and up to ~1068 s observed in the fairness-test log;
  repeated timing is impractical). Its spread is taken across the six cells.
- **Environment:** macOS-26.4 arm64 (Apple Silicon), Python 3.12.13, numpy 2.4.6, scipy 1.17.1, qutip 5.3.0.
  Tolerances: rtol=1e-5, atol=1e-7 (truncation rollouts); `run_from`/`run_v6` defaults (2nd-order/V7). Horizon
  t ∈ [0, 40], 200 points. Truncation Fock cutoff Nc = 10. Same machine for all methods.

## Method-by-method table (seconds; aggregated over the six M=3 cells)

| Method | build (median) | build (min–max) | rollout (median) | rollout (min–max) | total (median) | stability |
|---|---|---|---|---|---|---|
| 2nd-order | 3.04 | 3.02–3.14 | 0.020 | 0.019–0.022 | **3.06** | stable |
| V7 (learned) | 3.03 | 2.99–3.39 | 1.52 | 1.47–1.57 | **4.57** | stable |
| κ4=0 (T3) | 18.8 | 18.3–20.5 | 1.31 | 1.20–7.66 | **20.9** | stable at M=3 (ran to horizon) |
| κ5=0 (T4) | 76.1 | 72.5–89.8 | 5.19 | 4.2–88.0 | **81.2** | guard triggered (runaway growth) |

Ratios (medians): V7 total is ≈ **1.5×** the 2nd-order total, ≈ **4.5×** below κ4=0, and ≈ **18×** below κ5=0.
V7's **rollout** is ≈ **76×** the 2nd-order rollout (1.52 s vs 0.020 s).

## Caveats — T3/T4 instability and cost variability

- **κ4=0 (T3)** is stable at M=3 and ran to the horizon here; it becomes unstable in the low-loss pocket at M=4
  (fairness test), so its M=3 rollout time does not characterize its behaviour at larger M. One CONTROL cell
  showed a slower rollout (7.66 s) and a build-time spike (max 54.8 s), i.e. its cost is variable.
- **κ5=0 (T4)** does not run to the horizon: its rollout is timed only until the runaway-growth guard triggers
  (guard simulation-time t ≈ 12.06 CONTROL, 9.25 POCKET). Its wall-clock rollout is highly variable (4.2–88.0 s
  here, up to ~1056 s observed in the fairness-test log) because the solver crawls near the runaway. Build time
  is also variable (72–90 s here; up to ~1068 s observed elsewhere). All T4 rows are 1 run/cell.
- The build for 2nd-order and V7 is the *same* scalable engine; V7 adds the closure applier. V7's total advantage
  over T3/T4 is therefore **build-dominated** — it avoids constructing a higher-order engine.
- All numbers are at **M=3 only**, on one machine; no build-time scaling across M was measured. Nothing here
  claims behaviour at larger M or on other hardware.

## One conservative sentence (safe for the manuscript)

> In the evaluated M=3 cells, V7's median total wall-clock time (≈4.6 s) was close to the second-order closure
> (≈3.1 s) and roughly 4× and 18× below the tested κ4=0 and κ5=0 truncations (≈21 s and ≈81 s), because V7
> reuses the second-order engine and adds a learned per-step evaluation rather than constructing a higher-order
> engine; its per-rollout time (≈1.5 s) is substantially above the second-order rollout (≈0.02 s).

## One sentence that would be overclaiming (do NOT use)

> ❌ "V7 runs at essentially second-order cost — a cheap, fast closure with near-second-order runtime."

This is false: V7's per-rollout cost is ≈76× the second-order rollout, the comparison is at M=3 only on one
machine, and "cheap/fast" are not supported as absolute claims.
