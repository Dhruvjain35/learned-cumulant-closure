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
| 1 | 2nd-order overshoots by ≈ +0.28 to +0.30, roughly M-flat over M=2–5, both regimes | `src/evaluation/burst_pred.py` | M=2,3,4,5; POCKET+CONTROL | 50, 51, 52 | `results/burst_pred_exact.log`, `results/burst_pred_exact.npz` | `3830b7b` |
| 2 | κ4=0 (3rd) undershoots by ≈ −0.25 to −0.28 | `src/evaluation/burst_t3.py`; `src/baselines/t4_eval.py` | M=2,3,4 (t3); M=3 (grid); POCKET+CONTROL | 50 (t3); 50,51,52 (grid) | `results/burst_t3.log`, `results/t4_eval.log` | `b53304f`, `a855609` |
| 3 | κ4=0 (3rd) shows runaway growth triggering the guard in the low-loss pocket at M=4 | `src/evaluation/burst_t3.py` | M=4; POCKET | 50 | `results/burst_t3.log` (POCKET M=4 → UNSTABLE/failed) | `b53304f` |
| 4 | κ5=0 (4th) overshoots the rise, then reproducible runaway growth triggering the divergence guard; matched across four solvers (LSODA rtol 1e-5 & 1e-8, Radau, BDF), guard time t ≈ 9.65 | `src/baselines/t4_diag.py` | M=3; POCKET | 50 | `results/t4_diag.log` | `6228436` |
| 5 | κ5=0 (4th) triggers the guard in all six evaluated M=3 cells | `src/baselines/t4_eval.py` | M=3; POCKET+CONTROL | 50, 51, 52 | `results/t4_eval.log`, `results/t4_eval.npz` | `a855609` |
| 6 | κ5=0 engine assembly gate: constructed derivatives match the exact generator to ≈ 1e-14; κ5=0 closure reproduces κ4=0 coefficients at order 4 | `src/baselines/t4_truncation.py` (`__main__`) | M=3; κ=0.03, γφ=0.02, σ=0.15 | 5 (gate cell) | `results/t4_gate.log` | `1c547c1` |
| 7 | κ5=0 construction cost substantial and highly variable at M=3 (build 72–1068 s; 348 tracked moments; failed rollouts up to ~1056 s) | `src/baselines/t4_eval.py` | M=3; POCKET+CONTROL | 50, 51, 52 | `results/t4_eval.log` | `a855609` |
| 8 | V7 reproduces the burst peak to ≈ ±0.7% in the exact-validated cases | `src/evaluation/burst_pred.py`; `src/baselines/t4_eval.py` | M=2–5 (burst_pred); M=3 (grid); POCKET+CONTROL | 50, 51, 52 | `results/burst_pred_exact.log`, `results/t4_eval.log` | `3830b7b`, `a855609` |
| 9 | Baseline burst-trace figure (exact / 2nd / κ4=0 / κ5=0-to-guard / V7) | export: `scripts/export_baseline_burst_trace.py` (frozen models, deterministic); plot: `scripts/plot_baseline_burst_trace.py` (reads saved data only) | M=3; POCKET | 50 | `results/baseline_burst_trace_M3_pocket_draw50.{npz,png,pdf}` | `1c547c1` |

Notes:
- Rows 1, 4, 5, 7, 8 are the load-bearing quantitative claims; each maps to a committed log.
- The κ5=0 guard-trigger time is integrator-dependent: the multi-solver stability test (`t4_diag.log`) reports
  t ≈ 9.65; the single-solver trajectory export used for the figure reports t ≈ 9.25 (LSODA with `max_step=0.5`).
  Both trigger the guard; the runaway is consistent across these settings, not tied to one integrator.
- `1c547c1` = the packaging-pass commit that adds `results/t4_gate.log` and the burst-trace artifacts
  (filled in after commit; see `git log --oneline -- results/t4_gate.log`).
- No memory measurements exist in the repository; no runtime is claimed for V7 (see the section and the poster
  do-not-use list).
