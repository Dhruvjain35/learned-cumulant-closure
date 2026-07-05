# Replication quickstart

Goal: a collaborator who did not write this project clones the repo, installs dependencies, runs one command,
and determines whether they reproduce the reported **second-order cumulant burst-peak overshoot** against an
exact master-equation solve — in an independent codebase (Julia: QuantumCumulants.jl + QuantumOptics.jl).

This checks a reproduction. It does **not** assert the Julia implementation is validated; that judgement is
yours to record in `REPLICATION_SIGNOFF.md` after you run it and read `REPLICATION_AUDIT.md`.

---

## 1. What you need

- **Julia ≥ 1.9** (for `pkgversion`/`Pkg.dependencies`). Install via [juliaup](https://github.com/JuliaLang/juliaup)
  or the [official downloads](https://julialang.org/downloads/).
- Internet access on first run (to install packages from the Julia registry).
- The repo, cloned, with `results/replication_params.json` present (it ships in the repo).

The exact parameters — including the per-emitter disorder draws `w_i` — are read from
`results/replication_params.json`, so this is an exact-parameter cross-check, not a qualitative one.

## 2. Install dependencies (one time)

From the repository root:

```bash
julia --project=. julia_setup.jl
```

This activates a project environment in the repo folder and installs QuantumCumulants, QuantumOptics,
OrdinaryDiffEq, ModelingToolkit, and JSON (SHA is a standard library). It resolves package versions from the
registry — **record the versions it prints** in `REPLICATION_SIGNOFF.md`.

> The author did not pin exact package versions (no Julia runtime was available to author the script). If a
> newer package version changes an API the script calls, that is expected drift — see `REPLICATION_AUDIT.md`.

## 3. Run

```bash
julia --project=. replicate_cumulant.jl
```

## 4. Expected runtime

- **First run: ~5–20 minutes.** Most of it is Julia precompilation + the symbolic cumulant derivation
  (`complete(...)`) at M=4, plus the exact master-equation solve (density matrix of dimension 16·2^M).
- Subsequent runs are faster (precompilation cached). If M=4 is slow on your machine, editing `main()` to run
  only `M in (2, 3)` still exercises the full comparison.

## 5. Expected output

- Console: a table with columns
  `regime | M | F_peak_exact | F_peak_cumulant | signed_relerr | t_peak_ex | t_peak_cum | disorder_hash`,
  preceded by an environment/version block and the parameter-file sha256.
- File: **`results/replication_cumulant.csv`** with one row per cell (seed 50, M = 2, 3, 4, both regimes),
  including `params_sha256` and `disorder_hash` for provenance.

Reference targets from the project's Python engines (exact QuTiP `mesolve`, Fock cutoff 16; second-order
cumulant engine == dense engine to ~1e-14), **signed** relative burst-peak error
`(F_peak_cumulant − F_peak_exact) / F_peak_exact`, seed 50:

| regime  | M=2 | M=3 | M=4 | exact n_peak (M=2,3,4) |
|---------|-----|-----|-----|------------------------|
| POCKET  (κ=0.03) | +0.278 | +0.284 | +0.290 | 1.624, 2.312, 2.959 |
| CONTROL (κ=0.15) | +0.290 | +0.289 | +0.296 | 1.411, 1.996, 2.538 |

(`F_peak = κ · n_peak`, so POCKET F_peak ≈ 0.049/0.069/0.089, CONTROL ≈ 0.212/0.299/0.381.)

## 6. Success / failure criteria

**Pass (reproduction):** the second-order cumulant **overshoots** the burst peak — `signed_relerr` is
**positive, roughly +0.25 to +0.35, in both regimes**, and approximately flat across M. Matching the reference
`n_peak` column to solver tolerance is a bonus (you share the same disorder draws via the JSON), not required.

**Does not reproduce:** `signed_relerr` is negative, near zero, or far outside +0.25..+0.35 in the cells that
ran cleanly. Record this outcome exactly as it is — a non-reproduction is a valid, useful result.

**Inconclusive:** the script halts on an assertion (see §7). Fix the flagged cause, or record it as a blocker.

## 7. Known limitations and common errors

- **Not execution-verified by the author.** No Julia runtime was available when the script was written.
- **The cumulant initial-condition mapping is the fragile part.** `product_state_value` maps each tracked
  average to its all-excited product-state value using string matching on operator names, which depends on
  QuantumCumulants' printing. The script **guards** this: it halts with
  `initial <a'a> != 0 ...` or `initial total excited pop != M ...` if the mapping is wrong for your version.
  If you hit that, fix the mapping (guidance in `REPLICATION_AUDIT.md`) — do not trust the numbers until the
  two initial-condition checks pass.
- **API drift.** `meanfield`, `complete`, `ODESystem`, and `eqs.states` names/behaviour can change across
  QuantumCumulants/ModelingToolkit versions. If the script errors there, adapt the call and note it in the audit.
- **Fock cutoff.** The exact solve uses `NCUT = 16`. The script halts if the burst reaches ≥ 45% of the cutoff
  (would be truncation-biased); raise `NCUT` and rerun if so. For these cells n_peak ≲ 3, well within 16.
- **Cost grows with M.** The symbolic derivation and the exact Hilbert space both grow with M; M ≤ 4 is intended.

When done, fill in `REPLICATION_SIGNOFF.md` and send it back. Thank you.
