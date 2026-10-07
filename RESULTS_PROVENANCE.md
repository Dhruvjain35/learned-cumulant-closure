# Results provenance — Baselines, Failure Modes, and Practical Necessity section

For each final claim in that manuscript section: the generating script, the input parameters, the disorder
draw(s), the output file(s) that hold the evidence, and the commit hash of that evidence in this repository.

Common model and settings (all rows unless noted): open disordered Tavis–Cummings, RWA; ωc = 1; γ↓ = 0.02;
disorder σ = 0.15 (Gaussian about ωc); fully-inverted initial state. **POCKET (long-coherence)** =
(g_c, κ, γφ) = (0.70, 0.03, 0.02); **CONTROL** = (0.70, 0.15, 0.02). Exact reference: QuTiP `mesolve`, Fock
cutoff Nc = 16. Truncation engines (κ4=0, κ5=0): Nc = 10, Fsafe = 6. All disorder draws are deterministic
(seeded), so the values are reproducible on re-run.

| # | Claim (as stated in the section) | Script | Parameters | Disorder draw(s) | Output file(s) | Commit |
|---|---|---|---|---|---|---|
| 1 | 2nd-order overshoots by ≈ +0.28 to +0.30, roughly M-flat over M=2–5, both regimes | `src/evaluation/burst_pred.py` | M=2,3,4,5; POCKET+CONTROL | 50, 51, 52 | `results/burst_pred_exact.log`, `results/burst_pred_exact.npz` | `edfc9e1` |
| 2 | κ4=0 (3rd) undershoots by ≈ −0.25 to −0.28 | `src/evaluation/burst_t3.py`; `src/baselines/t4_eval.py` | M=2,3,4 (t3); M=3 (grid); POCKET+CONTROL | 50 (t3); 50,51,52 (grid) | `results/burst_t3.log`, `results/t4_eval.log` | `137a85d`, `323cb45` |
| 3 | κ4=0 (3rd) shows runaway growth triggering the guard in the low-loss pocket at M=4 | `src/evaluation/burst_t3.py` | M=4; POCKET | 50 | `results/burst_t3.log` (POCKET M=4 → UNSTABLE/failed) | `137a85d` |
| 4 | κ5=0 (4th) overshoots the rise, then reproducible runaway growth triggering the divergence guard; matched across four solvers (LSODA rtol 1e-5 & 1e-8, Radau, BDF), guard time t ≈ 9.65 | `src/baselines/t4_diag.py` | M=3; POCKET | 50 | `results/t4_diag.log` | `98b045d` |
| 5 | κ5=0 (4th) triggers the guard in all six evaluated M=3 cells | `src/baselines/t4_eval.py` | M=3; POCKET+CONTROL | 50, 51, 52 | `results/t4_eval.log`, `results/t4_eval.npz` | `323cb45` |
| 6 | κ5=0 engine assembly gate: constructed derivatives match the exact generator to ≈ 1e-14; κ5=0 closure reproduces κ4=0 coefficients at order 4 | `src/baselines/t4_truncation.py` (`__main__`) | M=3; κ=0.03, γφ=0.02, σ=0.15 | 5 (gate cell) | `results/t4_gate.log` | `130cf44` |
| 7 | κ5=0 construction cost substantial and highly variable at M=3 (build 72–1068 s; 348 tracked moments; failed rollouts up to ~1056 s) | `src/baselines/t4_eval.py` | M=3; POCKET+CONTROL | 50, 51, 52 | `results/t4_eval.log` | `323cb45` |
| 8 | V7 reproduces the burst peak to ≈ ±0.7% in the exact-validated cases | `src/evaluation/burst_pred.py`; `src/baselines/t4_eval.py` | M=2–5 (burst_pred); M=3 (grid); POCKET+CONTROL | 50, 51, 52 | `results/burst_pred_exact.log`, `results/t4_eval.log` | `edfc9e1`, `323cb45` |
| 9 | Baseline burst-trace figure (exact / 2nd / κ4=0 / κ5=0-to-guard / V7) | export: `scripts/export_baseline_burst_trace.py` (frozen models, deterministic); plot: `scripts/plot_baseline_burst_trace.py` (reads saved data only) | M=3; POCKET | 50 | `results/baseline_burst_trace_M3_pocket_draw50.{npz,png,pdf}` | `130cf44` |

Notes:
- Rows 1, 4, 5, 7, 8 are the load-bearing quantitative claims; each maps to a committed log.
- The κ5=0 guard-trigger time is integrator-dependent: the multi-solver stability test (`t4_diag.log`) reports
  t ≈ 9.65; the single-solver trajectory export used for the figure reports t ≈ 9.25 (LSODA with `max_step=0.5`).
  Both trigger the guard; the runaway is consistent across these settings, not tied to one integrator.
- `130cf44` = the packaging-pass commit that adds `results/t4_gate.log` and the burst-trace artifacts
  (filled in after commit; see `git log --oneline -- results/t4_gate.log`).

## External reproduction of row 1

Row 1 (the second-order burst-peak overshoot) is the one claim that has been reproduced outside this
repository. An outside collaborator ran `replicate_cumulant.jl` — an independent implementation in Julia
(QuantumCumulants.jl + QuantumOptics.jl) rather than the Python engines used above — at the Notre Dame CRC on
2026-08-30, under RHEL 9.8 / Julia 1.11.2 / QuantumCumulants 0.7.0.

| Claim | Artifacts | Verified on receipt |
|---|---|---|
| 2nd-order overshoots the burst peak by ≈ +0.28 to +0.30, M-flat, both regimes | `external_validation/{REPLICATION_SIGNOFF.md, replication_cumulant.csv, replication_run.log, replicate_cumulant_as_run.jl}` | parameter-file SHA256 byte-identical to `results/replication_params.json`; as-run script equals `replicate_cumulant.jl` at `b309fe8` plus three print-only lines; fail-closed IC guards logged as passed in all six cells; returned exact peaks match the Python reference to < 4e-4 |

## References above the 2^M wall (Results 34-37)

Added when the V8 line built references at sizes the rows above could not reach.

| Claim | Script | Reference method | Output |
|---|---|---|---|
| Burst-peak error of both closures at M = 5..16, zero disorder | `src/evaluation/v8_accuracy_ladder.py`, `v8_accuracy_m16.py` | permutation-invariant exact (`qutip.piqs`), gated to 1e-8 vs `mesolve` at M = 2..5 | `results/v8_accuracy_ladder.log`, `v8_accuracy_M16.log` |
| Burst-peak error at M = 12 **with disorder** | `src/evaluation/v8_disordered_m12.py` | quantum trajectories, 400 trajectories, ~1% MC error | `results/v8_disordered_m12.log` |
| Monte-Carlo noise floor justifying that trajectory count | `src/evaluation/v8_mc_noise_floor.py` | — | `results/v8_mc_noise_floor.log` |
| Physicality of the frozen closure at M = 5..20, zero disorder | `src/evaluation/v8_step0_sigma0.py`, `v8_forced_branch.py` | reference-free bounds | `results/v8_step0_sigma0.log`, `v8_forced_branch.log` |
| sigma-sensitivity at fixed M = 5 | `src/evaluation/v8_sigma_sensitivity.py` | exact `mesolve` | `results/v8_sigma_sensitivity.log` |
| Tensor networks do not reach these sizes | `src/baselines/v8_mps_feasibility.py` | TeNPy (isolated venv; not in the pinned lock file) | `results/v8_mps_feasibility.log`, `v8_mps_gate_reproduction.log` |

This is a cross-platform reproduction, not an independent replication. See
`external_validation/PROVENANCE.md` for the full receipt-side audit and
`docs/EXTERNAL_VALIDATION_STATUS.md` for what wording it licenses.
- No memory measurements exist in the repository; no runtime is claimed for V7 *in this baselines/necessity
  section* (see the section and the poster do-not-use list). V7 runtime is characterized separately in the
  frozen benchmark `results/runtime_baseline_comparison.md`, which reports a build-dominated total-time
  comparison (V7 total ≈4.6 s), not a speed claim.

## Common-carrier (rotating-frame) audit (Result 39)

Pre-registration: `docs/PROTOCOL.md` "Publication finalization P2", committed alone as `78e9a38` before any
shifted run. Results commit: the commit that adds `results/carrier_invariance.log`
(`git log --oneline -- results/carrier_invariance.log`).

| Claim | Script | Parameters | Draw | Output | Pass criterion (pre-registered) |
|---|---|---|---|---|---|
| Exact observables invariant under omega_c, w_i -> +C (n, z, peak, t_peak, `<a's-_1>` to <= 3e-13; charged moments identically 0) | `src/validation/check_carrier_invariance.py` | POCKET (0.70, 0.03, 0.02), CONTROL (0.70, 0.15, 0.02); M = 2, 3; C in {0, 9, 99, 999}; gdown 0.02, sigma 0.15; exact Nc = 16, mesolve adams rtol 1e-6 / atol 1e-8, nsteps 1e6 | 50 | `results/carrier_invariance.log`, `.npz` | rel <= 1e-5; charged <= 1e-7 |
| EOM coefficients shift by exactly i C Q (scalable <= 2.3e-13; dense M=2 <= 7.0e-11) | same | same | 50 | same | <= 1e-10 max(1, C) |
| Plain and V7 invariant only after the tolerance diagnostic (frozen tol: plain n <= 7.6e-4, V7 n <= 2.1e-3, V7 burst-peak error change <= 1.1e-3; RK45 rtol 1e-8: plain <= 5.1e-8, V7 <= 2.5e-6); verdict PASS AFTER TOLERANCE DIAGNOSTIC | same | closures Nc = 12, Fsafe = 6, RK45 rtol 1e-4 / atol 1e-6 (diagnostic 1e-8 / 1e-10) | 50 | same | rel <= 1e-4 (benchmark change <= 1e-4); exceedance must fall >= 10x at tight tolerance to count as noise |
| Frozen omega_c = 1 runs are bit-identical to omega_c = 0 rotating-frame runs (plain, V7: 0.0; exact <= 2.3e-14); charge-violating EOM residue <= 1.4e-14 (POST-HOC, not pre-registered) | `src/validation/check_rotating_frame_identity.py` | same four cells, frozen tolerances | 50 | `results/rotating_frame_identity.log` | none (post-hoc) |

The audit takes ~6.6 h on the benchmark machine, almost all of it in the omega_c = 1000 closure runs, where
explicit RK45 is limited by the fast-rotating charged components. C = 0 rows reproduce
`results/burst_pred_exact.npz` (draw 50) to 0.0 relative.

## Dated notes (2026-10-07)

- **"V8 line" (above) is an internal label** for the large-M reference audits of the frozen V7: the
  permutation-invariant, trajectory and MPS references, and the physicality studies. No V8 model exists. The only
  V8 model step ("V8 Step 1", a positivity-preserving closure) is a registered option that has **not been
  started** (PROTOCOL, hard stop 2026-12-31).
- **External reproduction precision, by quantity.** The returned CSV
  (`external_validation/replication_cumulant.csv`) against `results/burst_pred_exact.npz`
  (draw 50) at full precision:
  - exact F_peak: ≤ 2.7e-6 relative;
  - second-order F_peak: ≤ 7.6e-5 relative;
  - signed overshoot: ≤ 1.0e-4.
  The "< 4e-4" in row 1's external table is limited by a four-significant-figure reference.
- **Unarchived development check.** The 8.11e-10 Julia/Python closure agreement quoted in
  `REPLICATION_QUICKSTART.md` §7 comes from `scripts/crosscheck_negativity.{jl,py}`, run by the author at a
  strong-coupling point outside the evaluated cells. Only the Julia output (`results/crosscheck_negativity_julia.csv`)
  is committed. It has no provenance row and carries no publication weight.
- **Pre-registration timing.** Rows 1–8 above (Results 24 and 27) are not verifiably pre-registered; see
  `docs/HISTORY_PROVENANCE.md`.

## Held-out-parameter evaluation (Result 40) and M = 12 violation timing (Result 41)

- **Pre-registration:** PROTOCOL "Publication finalization P3", commit `34178e7`, committed before any run.
- **Results commit:** the commit that adds `results/heldout_param_burst.log`
  (`git log --oneline -- results/heldout_param_burst.log`).
- **Frozen models:** sha256 asserted by the script before any solve.
  - `lc_v3dens.pkl`: `be2dde48d47be006d97b7aa09c89f7a13e91dc2800fb6dc5cd6ea151fe798f81`
  - `lc_v6.pkl`: `0599643ef05f355f7549e38325261246f0f848fdc71d906f9f0f82b88b800b86`
- **Solver settings:**
  - Exact: QuTiP `mesolve` via `burst_pred.exact_ref_big`, Nc = 16 (fallback Nc = 24 if nc_peak/Nc ≥ 0.45;
    never triggered), t ≤ 40, 200 points.
  - Closures: `pilot_eval.build_cell` (ScalableEngine Nc = 12, Fsafe = 6), RK45 rtol 1e-4 / atol 1e-6.

| Claim | Script | Parameters | Draws | Output | Criterion (pre-registered) |
|---|---|---|---|---|---|
| Frozen V7 beats second order in 30/30 held-out-parameter cells at M = 5; V7 median \|err\| 0.29% (B), 0.45% (C), 1.05% (D); max 1.46%; second order 28.5–33.1%; V7 exactly physical in all 30 | `src/evaluation/heldout_parameter_burst.py` (guards: model hashes, `training_support.classify`, seed disjointness) | 30 cells in `results/heldout_param_cells.json` (12 B, 12 C, 6 D; 17 V4, 13 V6), M = 5, γ↓ = 0.02, fully inverted IC, Gaussian disorder | 300–329, one per cell | `results/heldout_param_burst.log`, `.npz` | Bands in PROTOCOL P3 → **STRONG TRANSFER** |
| Training support (81 tuples) and the per-cell class | `src/evaluation/training_support.py` (grid imported from `gen_data_v3.py`, `gen_pocket_density.py`; self-test asserts CONTROL = A, POCKET = C) | — | — | `results/heldout_param_cells.json` | — |
| 7 of 30 cells lie outside the trained per-axis ranges in ratio-feature coordinates (POST-HOC) | `src/evaluation/heldout_feature_support.py` | same | — | `results/heldout_feature_support.log` | none (post-hoc) |
| Result 37 disordered M = 12 cells stay physical to t = 40; burst peak t = 4.02 is the whole-run maximum; Result 37 errors reproduced (EXPLORATORY) | `src/evaluation/m12_violation_timing.py` | M = 12, σ = 0.15, POCKET and CONTROL, t ≤ 40 | 50 | `results/m12_violation_timing.log` | none (exploratory) |
