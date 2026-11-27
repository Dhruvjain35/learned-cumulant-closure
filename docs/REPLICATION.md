# Independent replication protocol — open disordered Tavis-Cummings, superradiant burst

> **Replication package (start here):**
> - [`REPLICATION_QUICKSTART.md`](REPLICATION_QUICKSTART.md) — install, one command to run, expected output, pass/fail criteria.
> - `REPLICATION_SIGNOFF.md` (not in the public release) — fill-in table to record environment, results, and verdict.
> - `REPLICATION_AUDIT.md` (not in the public release) — line-by-line model/convention audit, assumptions, mismatch risks, questions to check.
> - [`replicate_cumulant.jl`](../replicate_cumulant.jl) — the reproduction script (Julia). Execution-verified against the pinned manifests; reproduced cross-platform by an outside collaborator on 2026-08-30 (`external_validation/`).
> - [`results/replication_params.json`](../results/replication_params.json) — exact parameters incl. disorder draws.
>
> This document gives the model spec and reproduction targets; the quickstart is the fastest path to running it.

Purpose: have someone **who did not write the Python code** independently confirm, in a **separate codebase**
(QuantumCumulants.jl + QuantumOptics.jl), the two claims the rest of this project is measured against:

1. the **exact** reference `n(t) = ⟨a†a⟩(t)` for the open disordered Tavis-Cummings model, fully-inverted IC;
2. that the **standard 2nd-order cumulant closure overshoots the superradiant burst peak** by ~29%.

The learned closure (V7) is **not** part of this replication — it is the model under test. The point of the
replication is to establish the **baseline** (exact solve + plain cumulant) outside this codebase, so that V7's
measured advantage rests on a reference that has been reproduced cross-platform, not on our own engine alone.
Status (2026-10-07): the cross-platform tier is done; the independent-replication tier below is not.

## Model (exact spec)

- Hamiltonian (RWA), rotating nothing, `ωc = 1`:
  `H = ωc a†a + Σ_i w_i σ_i⁺σ_i⁻ + Σ_i g_i (a σ_i⁺ + a† σ_i⁻)`, with `g_i = g_c/√M` (collective coupling
  `g_c` is M-independent by construction — this matters, see "What NOT to expect").
- Dissipators (Lindblad): `κ D[a]` + per emitter `γ↓ D[σ_i⁻]` + `(γφ/2) D[σ_i^z]`.
- Convention: `σ^z = |e⟩⟨e| − |g⟩⟨g|`, `σ⁺ = |e⟩⟨g|`, `σ⁻ = |g⟩⟨e|`. Fixed `γ↓ = 0.02`, disorder `σ = 0.15`
  (Gaussian about `ωc`).
- Initial state: **cavity vacuum, every emitter excited** (fully inverted / superradiant IC).
- Observable: emitted flux `κ⟨a†a⟩(t)`; burst peak `F_peak = κ·max_t⟨a†a⟩`, delay `t_peak = argmax_t⟨a†a⟩`.
  Time grid `t ∈ [0, 40]`, 200 points.

## Exact parameters (turnkey cross-check)

`results/replication_params.json` holds every cell's exact parameters **including the disorder draws `w_i`**,
so the independent run uses identical detunings and couplings. `replicate_cumulant.jl` reads it. Two regimes:

- **POCKET**  `g_c=0.70, κ=0.03, γφ=0.02` (long-coherence; where 2nd-order cumulant is known to fail here)
- **CONTROL** `g_c=0.70, κ=0.15, γφ=0.02` (normal loss; negative control)

To instead test physical robustness rather than bit-match, draw your own Gaussian `w_i` (`σ=0.15`, `ωc=1`).

## What to reproduce (targets, seed 50)

Our validated Python engines (exact QuTiP `mesolve`, Fock cutoff 16; plain 2nd-order cumulant == dense engine
to 1e-14) give, for the **signed** relative burst-peak error of plain cumulant vs exact,
`(F_peak^cumulant − F_peak^exact)/F_peak^exact`:

| regime  | M=2 | M=3 | M=4 | M=5 | exact n_peak (M=2..5) |
|---------|-----|-----|-----|-----|-----------------------|
| POCKET  | +0.278 | +0.284 | +0.290 | +0.291 | 1.624, 2.312, 2.959, 3.583 |
| CONTROL | +0.290 | +0.289 | +0.296 | +0.303 | 1.411, 1.996, 2.538, 3.053 |

**Acceptance criterion (physical, not bit-exact):** an independent QuantumCumulants.jl 2nd-order solve should
reproduce a **positive burst-peak overshoot of order +25% to +35%**, roughly M-flat, in **both** regimes, and
an exact QuantumOptics.jl `⟨a†a⟩(t)` that matches the `n_peak` column to solver tolerance. Reproducing the
overshoot is the pass condition; matching floats to 3 digits is a bonus (disorder draws are shared via JSON).

## What NOT to expect (already established, report honestly)

- The overshoot is **flat in M**, not growing — because `g_c` is held M-independent, so the collective burst
  is ~M-independent by construction. Do not expect (or claim) a superradiant N-scaling in this convention.
- It is **not** pocket-specific — plain cumulant overshoots in the control regime too. This is a general
  property of 2nd-order cumulant on the inverted IC, not a memory-routing effect.

## Files

- `replicate_cumulant.jl` — QuantumCumulants.jl + QuantumOptics.jl script (reads the JSON). *Originally written
  without a Julia runtime; since 2026-08-16 it is execution-verified by the author against the committed
  `Manifest.toml` / `Manifest-v1.11.toml`, and on 2026-08-30 it was reproduced cross-platform by an outside
  collaborator (`external_validation/`).* The initial-value mapping remains the most
  version-sensitive part (see `REPLICATION_QUICKSTART.md` §7).
- `results/replication_params.json` — exact per-cell parameters incl. disorder.
- Output: `results/replication_cumulant.csv` (regime, M, F_peak exact vs cumulant, rel err, t_peak).

## Tiny hand-check (M=2, fully transparent)

At M=2 the 2nd-order cumulant system is small enough to write out and integrate by hand or in a few lines of
independent code (no package). Reproducing the +0.28 pocket overshoot at M=2 with a from-scratch integrator is
the most transparent triple-check and is poster-friendly (the equations fit on a slide).

## Claim tiers (wording discipline — do not conflate)

- **Cross-platform reproduction** = an independent person runs *this* artifact with the provided JSON and
  reproduces the ~29% second-order burst-peak overshoot. Valuable, but it is running *my* code in another
  language — not fully independent. Poster wording: "cross-platform reproduction in QuantumCumulants.jl."
- **Independent replication** = the above **plus**: (i) an independent check of the EOM / sign conventions,
  (ii) at least one M=2 case verified against the replicator's *own* derivation or a *separately assembled*
  QuantumCumulants setup, (iii) recorded Julia version, package versions, OS/hardware, commit hash,
  parameters, and the numerical difference. Only this tier earns: **"A cross-platform implementation in
  QuantumCumulants.jl independently reproduced the exact-vs-cumulant burst-peak discrepancy within the stated
  tolerance."** Until then, say "reproduced cross-platform in QuantumCumulants.jl" (done 2026-08-30), not "independently replicated."

Sign-off record (replicator fills in): code/commit, Julia + package versions, OS/hardware, reproduced
overshoot table, confirmation `⟨a†a⟩(t)` matches, sign/direction match, name + date.

## Request to send the replicator (verbatim)

> Could you independently validate two parts of my project?
>
> **1. Cross-platform reproduction.** Please run the attached QuantumCumulants.jl / QuantumOptics.jl artifact
> using the provided JSON parameters and verify whether it reproduces: exact dynamics, second-order cumulant
> dynamics, and the approximately 28–30% second-order burst-peak overshoot. Please record: Julia version,
> package versions, OS/hardware, peak values and relative difference, and whether the sign/direction of the
> discrepancy matches.
>
> **2. Theory sanity check.** Please review my inverted-state linearization for the open Tavis-Cummings bright
> mode: gamma_perp = gamma_down/2 + gamma_phi, and the collective-mode eigenvalue derivation. I specifically
> need you to check: sign conventions in the alpha/P equations; whether the inverted state produces real gain
> eigenvalues rather than vacuum-Rabi oscillations; the gamma_perp definition; and whether the approximation
> is valid only on pre-burst timescales because the inversion decays.
>
> I am not asking you to improve the model or add new experiments — only to identify any mathematical or
> implementation mistake.
