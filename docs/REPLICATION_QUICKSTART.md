# Replication quickstart

Goal: a collaborator who did not write this project clones the repo, installs dependencies, runs one command,
and determines whether they reproduce the reported **second-order cumulant burst-peak overshoot** against an
exact master-equation solve — in an independent codebase (Julia: QuantumCumulants.jl + QuantumOptics.jl).

This checks a reproduction. It does **not** assert the Julia implementation is validated; that judgement is
yours to record in `REPLICATION_SIGNOFF.md` after you run it and read `REPLICATION_AUDIT.md`.

---

## 1. What you need

- **Julia 1.12.6 recommended** — that is the version `Manifest.toml` was generated on, so it is the version the
  pinned environment instantiates cleanly against. Install via
  [juliaup](https://github.com/JuliaLang/juliaup) (`juliaup add 1.12.6`) or the
  [official downloads](https://julialang.org/downloads/). Julia ≥ 1.10 should work via the `[compat]` fallback
  described in §2, but a different minor version may fail to instantiate the Manifest.
- Internet access on first run (to install packages from the Julia registry).
- The repo, cloned, with `results/replication_params.json` present (it ships in the repo).

The exact parameters — including the per-emitter disorder draws `w_i` — are read from
`results/replication_params.json`, so this is an exact-parameter cross-check, not a qualitative one.

## 2. Install dependencies (one time)

From the repository root:

```bash
julia --project=. julia_setup.jl
```

This activates the project environment in the repo folder and **instantiates the exact pinned dependency set**
recorded in `Manifest.toml` — the set the script has been executed against:

| package | pinned version |
|---|---|
| Julia | 1.12.6 |
| QuantumCumulants | 0.7.0 |
| QuantumOptics | 1.2.7 |
| OrdinaryDiffEq | 7.6.0 |
| ModelingToolkitBase | 1.65.0 |
| SecondQuantizedAlgebra (transitive) | 0.10.0 |
| JSON | 1.7.1 |

**Record the versions it prints** in `REPLICATION_SIGNOFF.md` anyway — if they differ from this table, that is
itself worth knowing.

> **If `Pkg.instantiate()` fails**, it is almost certainly a Julia-version mismatch. The setup script prints
> your options: run on Julia 1.12.6, or delete `Manifest.toml` and re-run to resolve from the exact `[compat]`
> pins in `Project.toml` instead. Either way, report the failure — a resolution failure is useful information.

> **Earlier unpinned versions of this repo** resolved dependencies from the registry at run time, which caused
> genuine API-drift failures in the first external attempt. That is fixed; see §7.

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

- **Execution status.** The script **is** now execution-verified against the pinned versions in §2: it runs end
  to end and produces the acceptance numbers. It is **not** thereby externally validated — it was run by the
  project's author. `docs/EXTERNAL_VALIDATION_STATUS.md` is the source of truth for that distinction, and the
  cross-platform-reproduction row there stays **NOT DONE** until someone outside the project records a result.
- **Fixed since the first external attempt (2026-08-16).** All four were unpinned-dependency API drift, not
  physics; none changed the model, the parameters, or any reported number:
  1. `dagger` is exported by both QuantumOptics and SecondQuantizedAlgebra (via QuantumCumulants 0.7), so the
     bare call was ambiguous. Now qualified `QuantumOptics.dagger`.
  2. QuantumCumulants 0.7 builds on **ModelingToolkitBase**, not the ModelingToolkit wrapper. Importing the
     wrapper made `ODEProblem` fail with `im is present in the system but im is not an unknown`. The script now
     imports `@named, mtkcompile, ODEProblem, unknowns` from `ModelingToolkitBase` and uses
     `System` + `mtkcompile` + a `Dict` initial-condition map (MTK 11 rejects a positional `u0` vector).
  3. **The one that mattered.** QuantumCumulants 0.7 renders transitions with *Unicode subscript* level indices
     (`σ1₂₂`), so the old `occursin("22", …)` mapping matched nothing and returned `0.0` for **every** state — a
     silently un-inverted initial condition rather than a crash. Worse, the guard meant to catch exactly this
     failed *open*: the single-population index list came back empty, so the check was skipped. The mapping is
     now parsed **structurally**, factor by factor, it **throws** on any factor it does not recognise, and the
     guards **fail closed** (an empty match is an error, and the population count must equal M).
  4. `sol.retcode == :Success` never matches modern SciML, which returns `ReturnCode.Success` — that guard would
     have thrown a false failure on a perfectly good solve.
- **Observable extraction.** `⟨a'a⟩(t)` is now read via QuantumCumulants' own `get_solution(sol, a'*a, eqs)`
  rather than by positional index, and **cross-checked** against the directly-indexed unknown; the script halts
  if the two disagree by more than 1e-8.
- **Negative photon numbers outside the tested cells are expected, and are not a bug.** If you push the coupling
  well above the evaluated cells (e.g. `g = 0.5`), the second-order closure returns a *negative* `⟨a'a⟩(t)`.
  That is a positivity violation of the closure itself, not of this script. It is cross-checked in
  `scripts/crosscheck_negativity.{jl,py}`: at an identical strong-coupling point, on a shared time grid at
  matched tolerances, the Julia closure and the project's own Python engine agree to **8.11e-10** over all 201
  grid points while the exact `mesolve` reference stays non-negative. The evaluated cells (`g_c = 0.70`,
  i.e. `g = g_c/√M`) are far from this and show no such excursion.
- **Remaining fragility.** `product_state_value` is still the most version-sensitive part. It now fails loudly
  instead of silently, but if your QuantumCumulants version renders operators differently it *will* halt with
  `UNRECOGNIZED FACTOR '…'`. That is the intended behaviour — fix the mapping (guidance in
  `REPLICATION_AUDIT.md`) rather than working around it.
- **Fock cutoff.** The exact solve uses `NCUT = 16`. The script halts if the burst reaches ≥ 45% of the cutoff
  (would be truncation-biased); raise `NCUT` and rerun if so. For these cells n_peak ≲ 3, well within 16.
- **Cost grows with M.** The symbolic derivation and the exact Hilbert space both grow with M; M ≤ 4 is intended.

When done, fill in `REPLICATION_SIGNOFF.md` and send it back. Thank you.
