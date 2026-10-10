# Figure provenance: main manuscript figures

Every main figure is **render-only**. `scripts/build_manuscript_figures.py` reads committed artifacts in
`results/` (plus three committed constants: the training grid in `src/evaluation/training_support.py`, the router
in `portable_v7/router.json`, and the held-out cell list in `results/heldout_param_cells.json`). It runs no exact
solve, no closure rollout and no network evaluation. Logs are parsed with strict patterns and expected row counts
are asserted, so a changed or truncated artifact fails the build and cannot silently plot wrong numbers.

```
make figures            # = python run.py scripts/build_manuscript_figures.py   (~5 s)
```

Output: `figures/fig{1..5}_*.pdf` (vector, Type 42 embedded fonts, no timestamps), `.eps` (production format for
colour-online figures; same plotted data) and `.png` (200 dpi preview).
A rebuild from a deleted `figures/` directory is byte-identical in all ten files (checked 2026-10-07).
Shared style: `src/figures/pubstyle.py`.

**Encoding, fixed across all figures.**
- Method is shown by colour, line style and marker together:
  - exact: black, solid
  - V7: blue, solid, circle
  - second order: orange, dashed, square
  - κ4 = 0: aqua, dash-dot, triangle
  - κ5 = 0: yellow, dotted, diamond
- Regime is shown by fill: low loss (κ = 0.03) dark-filled, normal loss (κ = 0.15) light-filled (a pale tint of the
  same color with a dark edge; restyled 2026-10-09 to avoid open symbols per APS figure guidance).
- Mechanism families use neutral ink: Aaz dark-filled, a+z light-filled.
- Unstable or runaway outcomes are a categorical × in a shaded band. They are never drawn as a bar or a number.
- No p-values appear anywhere.
- The palette was checked with the dataviz validator: every check passes. Aqua and yellow fall below 3:1 contrast, so every series also carries a marker, a line style and a label, and the figures read in grayscale.

| Fig | Panel | Content | Input artifact(s) | Result | Prospective / exploratory |
|---|---|---|---|---|---|
| 1 | a | model schematic (no data) | none | — | — |
| 1 | b | closure problem schematic; "≈ +29%" from the second-order burst error | text only; the value is from Results 24/40 | 24, 40 | — |
| 1 | c | architecture schematic (no data) | none | 17–19 | — |
| 1 | d | router box; training (κ, γφ) projection; Result 40 cells | `portable_v7/router.json`, `src/evaluation/training_support.py` (TRAIN), `results/heldout_param_cells.json` | 19, 40 | router frozen |
| 2 | a | M = 3 low-loss draw 50 trace: exact / second order / V7 | `results/baseline_burst_trace_M3_pocket_draw50.npz` | 24, 27 | illustrative cell |
| 2 | b | burst error, M = 2–5, draws 50–52, both regimes | `results/burst_pred_exact.npz` | 24 | held-out disorder draws; parameters on the training grid |
| 2 | c | 30 held-out parameter cells by support class, raw points + median/IQR | `results/heldout_param_burst.npz` (`rows`) | 40 | prospective (P3) |
| 2 | (d) | removed 2026-10-10; the per-class summary is Table I of the manuscript | `results/heldout_param_burst.npz` (`stats`, `rows`); 7/30 ratio-coordinate caveat from `results/heldout_feature_support.log` | 40 | prospective |
| 3 | a | signed error vs M = 5, 8, 10, 12, 16, σ = 0, both regimes; disordered M = 12 overlay | `results/v8_accuracy_ladder.log`, `results/v8_accuracy_M16.log`, `results/v8_disordered_m12.log` | 34, 35, 36, 37 | exploratory size transfer |
| 3 | b | M = 12: σ = 0 permutation-invariant exact vs σ = 0.15 trajectories | `results/v8_accuracy_ladder.log`, `results/v8_disordered_m12.log` | 36, 37 | one disordered draw |
| 3 | c | V7 full-trajectory physicality, regime × M, physical / tested | `results/positivity_audit.npz` (low loss σ = 0.15, seeds 7–9), `results/v8_step0_sigma0.npz` (σ = 0 rows), `results/m12_violation_timing.log` (σ = 0.15 M = 12, seed 50, both regimes) | 30, 34, 41 | 41 exploratory |
| 3 | d | burst-peak time vs first violation; no-violation rows censored at t = 40 | `results/v8_peak_vs_violation.log`, `results/m12_violation_timing.log` | 34, 41 | 41 exploratory |
| 4 | a | structural pathway; family counts (19 / 15 charged / 4 neutral) derived from the committed family labels | `results/mechanism_m5.npz` (`cells[0]["share_all"]` keys) | 25, 42 | structural fact (not tested) |
| 4 | b | per-draw $D_f$ share, Aaz vs a+z, six draws per regime | `results/mechanism_m5.npz` (`cells`) | 42 | prospective (P5) |
| 4 | c | M = 4 aggregate vs M = 5 median Aaz share | `results/mechanism.log` (M = 4), `results/mechanism_m5.npz` (M = 5) | 25, 42 | 25 exploratory, 42 prospective |
| 4 | d | closed-loop ablation loss per draw; unstable shown as categorical × | `results/mechanism_m5.npz` (`cells`) | 42 | prospective |
| 5 | a | truncation ladder at M = 3, draws 50–52, both regimes; mean field categorical | `results/t4_eval.npz` (`res`); mean-field statement asserted from `results/pilot_eval.log` | 20, 27 | exploratory |
| 5 | b | build / rollout / total wall-clock medians | `results/runtime_baseline_comparison.csv` | runtime benchmark (RESULTS_PROVENANCE) | single machine, M = 3 |
| 5 | c | M = 3 low-loss draw 50 trace incl. κ4 = 0 and κ5 = 0; κ5 = 0 shown until the divergence guard | `results/baseline_burst_trace_M3_pocket_draw50.npz` (`meta.t4_stop`, `meta.t4_guard_triggered`) | 27 | illustrative cell |

## Notes and checks

- **Fig 2b.** The original M ≤ 5 benchmark sits on training-grid parameters. It holds out disorder realisations, not parameters. Parameter transfer is shown only in 2c (summary: Table I).
- **Fig 2c and Table I.** Support classes are defined in raw physical parameters. In the model's ratio-to-κ inputs, only 7 of 30 cells lie outside the trained ranges. The figure does not call these cells fully out of distribution.
- **Fig 3a.** Only measured points are plotted, M ≤ 16. There is no fit line, no extrapolation and no crossover estimate.
- **Fig 3c.** The low-loss σ = 0.15 M = 12 cell reads 3/4: Result 30 seeds 7–9 give 2/3, and Result 41 seed 50 adds 1/1. Normal-loss σ = 0.15 was measured only at M = 12 (Result 41). Every other cell in that row reads "not measured", not "physical".
- **Fig 4c.** Draws, statistic and scope differ between the two points: M = 4 is the aggregate over seeds 37–39 as reported in the Result 25 log, M = 5 is the median over seeds 500–505. Because of that the panel is read as "the split reproduces", not as a paired comparison. a+z is 1 − Aaz up to ~1e-15 (structural check in Result 42).
- **Fig 4d.** Low loss: Aaz ablation is unstable in 6/6 draws, and a+z in 5/6 (one finite at +0.279). Normal loss: everything is finite, Aaz ≈ 0.81 and a+z ≈ 0.31. No infinite value is drawn as a bar.
- **Fig 5b.** V7 rollout is ≈ 75× the second-order rollout (1.5 s vs 0.02 s). V7's total of 4.6 s is close to second order's 3.1 s because the shared engine build dominates. That is a statement about build cost, not a claim that V7 costs nothing.

## Comparison with the earlier figures (same data, regenerated)

| Earlier figure | New panel | Agreement |
|---|---|---|
| `results/runtime_benchmark_fig.png` | Fig 5b | identical medians: totals 3.1 / 4.6 / 20.9 / 81.2 s |
| `assets/img/mechanism.png` (a) | Fig 4c, M = 4 points | identical shares 0.67/0.33 (low loss) and 0.65/0.35 (normal loss) |
| `assets/img/mechanism.png` (c) | Fig 4d (now at M = 5) | **the earlier figure drew the low-loss ablation as solid bars capped at 1.2** ("destabilizes"), while its log row is mean finite loss +0.81 with 1 of 3 seeds critical. That encoding draws instability as a magnitude. The new figure replaces it with a per-draw categorical ×. |
| `results/baseline_burst_trace_M3_pocket_draw50.png` | Fig 2a, Fig 5c | same `.npz`; same curves |
| `assets/img/ladder.png` | not reused in main text | supplement (Result 20 NRMSE ladder, different metric) |

The earlier figures stay in the repository unchanged. They are referenced by the README and the findings log, and none were regenerated or overwritten.

## PRA restyle (2026-10-09)

Typography and markers only:
- every text ≥ 9 pt, i.e. capitals about 2.1 mm in STIX;
- panel letters 11 pt;
- no open symbols;
- larger markers;
- no transparency;
- EPS output added.

Printed scale in the REVTeX manuscript (width `\textwidth` = 510 pt, height capped at 0.85 `\textheight`) is at least
1.01, so all figure text prints at ≥ 9 pt.

Data identity before and after the restyle is verified by `scripts/figure_signature.py`. The comparison log is
`results/figure_style_regression.log`: every panel's plotted arrays are exactly equal. Captions changed only in the
words that name the marker encoding ("hollow" → "light-filled", "Blue rings" → "Light-blue circles").
