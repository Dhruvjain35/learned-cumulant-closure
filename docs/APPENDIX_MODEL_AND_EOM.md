# Appendix: model, conventions, and the second-order equations of motion

This appendix states the model in full and records which parts of it are machine-checked, by what
argument, and what is left for a human reader. It exists so that a reader can verify the derivation
without corresponding with the author, and so that the project's claims about "checked" and
"unchecked" point at specific, rerunnable gates.

## A.1 Model

An open, disordered Tavis–Cummings ensemble: one cavity mode coupled to `M` two-level emitters under
the rotating-wave approximation, with inhomogeneous emitter frequencies.

**Hamiltonian** (ħ = 1):

```
H  =  ω_c a†a  +  Σ_i w_i σ⁺_i σ⁻_i  +  Σ_i g_i ( a σ⁺_i  +  a† σ⁻_i )
```

**Lindblad dynamics:**

```
dρ/dt  =  −i[H, ρ]  +  κ D[a] ρ  +  Σ_i γ↓ D[σ⁻_i] ρ  +  Σ_i (γφ/2) D[σᶻ_i] ρ

D[O] ρ  =  O ρ O†  −  ½ { O† O , ρ }
```

**Disorder.** `w_i` is drawn once per cell from a Gaussian of width σ about ω_c, from a fixed seed, so
every reported number is reproducible on rerun.

**Coupling convention.** `g_i = g_c / √M`, which holds the collective coupling `G = g_c` fixed as `M`
varies. This is a **chosen benchmarking convention** adopted so that different `M` can be compared at
matched collective coupling. It is not a physical scaling law, and nothing here derives it from one. It is
also the reason the burst-peak error comes out flat in `M` rather than growing; see Result 24, where the
growing-with-M hypothesis was tested and rejected.

**Coupling regime, and what the RWA costs here.** The evaluated cells use `g_c = 0.7` at `ω_c = 1`, so
`g_c/ω_c = 0.7` and the per-emitter `g_i/ω_c ≈ 0.495` at `M = 2`. That is ultrastrong coupling by the
conventional threshold of `g/ω_c ≳ 0.1`, and in that regime neither the rotating-wave approximation nor
bare-mode local Lindblad dissipation is the physically correct description: counter-rotating terms and
dressed-state dissipation are both omitted.

This is deliberate and it does not weaken any claim made here, because every claim is **internal to one
defined generator**. The exact reference, the cumulant closures, and the learned closure all solve the
same RWA-Lindblad model, so the comparison between them is exact at every coupling. What is being measured
is how closure schemes behave across that model's dynamical regimes, not how a physical cavity behaves at
`g/ω_c = 0.7`. No experimental accuracy is claimed at these couplings, and none should be read in.

**Initial state.** Cavity vacuum ⊗ all emitters excited, a fully inverted product state. This is the
superradiant burst preparation, and it is the reason second-order cumulant closure is stressed: the
inverted state has no coherence for mean-field to build on, so the dynamics is driven entirely by
correlations that the closure must supply.

**Observable.** The burst peak `F_peak = κ · max_t ⟨a†a⟩(t)`, and its time `t_peak = argmax_t ⟨a†a⟩(t)`.

## A.2 Conventions, and the closed-form result each one is tested against

Conventions are the part of a model that readers most often have to take on faith. Here each one is a
measured consequence of the code, checked against a result derived outside this codebase. All gates
are in [`../src/validation/check_conventions.py`](../src/validation/check_conventions.py); run with
`python run.py src/validation/check_conventions.py`.

| Convention | Closed-form consequence | Gate | Measured |
|---|---|---|---|
| Excited state is the +1 eigenstate of σᶻ | ⟨σᶻ⟩ = +1 excited, −1 ground; H costs `w` on the excited state; the burst IC has ⟨a†a⟩ = 0 and total excited population `M` | C2 | exact |
| Cavity loss `κ D[a]` | ⟨a†a⟩ decays at `κ`; ⟨a⟩ at `κ/2` | C3 | 3e-12 |
| Emitter decay `γ↓ D[σ⁻]` | ⟨σ⁺σ⁻⟩ decays at `γ↓`; \|⟨σ⁻⟩\| at `γ↓/2` | C4 | 4e-11 |
| Pure dephasing `(γφ/2) D[σᶻ]` | \|⟨σ⁻⟩\| decays at `γφ`, and population is untouched | C5 | 6e-11 |
| The two together | \|⟨σ⁻⟩\| decays at **γ⊥ = γ↓/2 + γφ** | C6 | 6e-11 |
| Same conventions inside the closure engine | the engine's own generator reproduces all four rates | C7 | 1e-17 |

The `γ⊥ = γ↓/2 + γφ` row is the one that matters most. It is an input to the Result 23 linearization,
and until these gates existed it was an assumption stated in prose. It is now a measured property of
the model as implemented.

Gate C1 additionally rebuilds every operator and the Hamiltonian from the equations printed in §A.1,
in raw NumPy, and checks them against what `exact_tc.py` constructs. Agreement is exact (0.0
elementwise). This rules out a transcription error between the documented model and the code.

## A.3 Second-order cumulant closure

Write `⟨·⟩` for expectation values and let a monomial of order `n` be a product of `n` fundamental
factors drawn from `{a†, a, σ⁺_i, σ⁻_i, σᶻ_i}`. The exact hierarchy for the tracked moments is

```
d⟨O⟩/dt  =  ⟨ L†[O] ⟩,        L†[X] = i[H, X] + Σ_c ( c† X c − ½ { c†c , X } )
```

and for `O` of order ≤ 2 the right-hand side involves monomials of order 3. Second-order cumulant
closure sets the connected third cumulant to zero, i.e. for a third-order monomial with factors
`f₁ f₂ f₃`:

```
⟨f₁f₂f₃⟩  ≈  ⟨f₁⟩⟨f₂f₃⟩ + ⟨f₂⟩⟨f₁f₃⟩ + ⟨f₃⟩⟨f₁f₂⟩ − 2⟨f₁⟩⟨f₂⟩⟨f₃⟩
```

The learned closure (V7) replaces the "≈" by supplying a correction `κ₃` to that expression rather
than assuming it vanishes. Everything else in the equations is unchanged, which is why V7 reuses the
second-order engine rather than building a higher-order one.

## A.4 The tracked set at M = 2

For `M = 2` the closure tracks 33 moments (8 of order 1, 24 of order 2, plus the identity), and
closing them requires 40 third-order monomials. Written out:

**Order 1 (8):** `a`, `a†`, `σ⁺₁`, `σ⁻₁`, `σᶻ₁`, `σ⁺₂`, `σ⁻₂`, `σᶻ₂`

**Order 2 (24):** `a²`, `a†²`, `a†a`, `aσ⁺ᵢ`, `aσ⁻ᵢ`, `aσᶻᵢ`, `a†σ⁺ᵢ`, `a†σ⁻ᵢ`, `a†σᶻᵢ` (for i = 1, 2),
and the nine emitter–emitter products `σ^α₁σ^β₂` with α, β ∈ {+, −, z}

At `M = 3` this becomes 60 tracked moments and 112 third-order monomials.

Only two connected third-cumulant families can enter these equations with a nonzero value for the burst
initial state:
- the photon-number-inversion channel `⟨a†aσᶻ⟩`;
- the photon-polarization-inversion channel `⟨aσ⁺σᶻ⟩`.
Every charge-carrying family vanishes by excitation-number conservation, and the remaining neutral families do
not couple to the tracked moments. Their split in the whole-dynamics influence score D_f was:
- 65–67% / 33–35% exploratorily at M = 4 (Result 25);
- 71% / 29% (low loss) and 62% / 38% (normal loss) in the pre-registered M = 5 confirmation (Result 42).
See the manuscript, §7.

## A.5 The derivation is machine-checked

The equations of motion are not hand-transcribed. `CumulantEngine` builds `L†[O]` for each tracked
`O` directly from `H` and the jump operators and projects it onto the monomial basis. That projection
is then verified by an identity rather than a tolerance:

> Feed the engine the **exact** moments and the **exact** third cumulants, computed from a full
> density matrix. The closure approximation then vanishes identically, so the assembled derivative
> must equal `⟨L†[O]⟩` to machine precision.

[`../src/validation/check_eom_identity.py`](../src/validation/check_eom_identity.py) runs this on the
cells the paper reports — POCKET and CONTROL, `M` = 2 and 3, the disorder draws in
`results/replication_params.json` — at six times spanning the burst. All 72 checks pass at ≈ 1e-14 or
better; the log is `results/eom_identity.log`.

If the projection coefficients, the moment-to-cumulant assembly, or the operator algebra were wrong
anywhere in the tracked set, this identity would fail. It does not.

## A.6 What is checked, and what is not

Machine-checked and rerunnable:

- The operators and Hamiltonian match the equations in §A.1 (C1).
- Every dissipative convention matches a closed-form result, including `γ⊥ = γ↓/2 + γφ` (C2–C7).
- The coherent spectrum matches analytic limits: vacuum Rabi `ω_c ± g` at `M` = 1, collective
  splitting `2g√M` (`validate_model.py`, gates 1 and 2).
- The second-order equations of motion and the cumulant assembly are exact given `H` and the jump
  operators (`check_eom_identity.py`).
- The second-order closure is reproduced through a separate code path: QuantumCumulants.jl derives the
  second-order equations symbolically, without this project's engine. On the six evaluated cells, run by an
  outside collaborator (`external_validation/`), its burst peaks agree with the Python engine to within
  1e-4 in the signed overshoot (0.01 percentage points).
  A development cross-check at a strong-coupling point outside the evaluated cells
  (`scripts/crosscheck_negativity.{jl,py}`, run by the author) agreed to 8e-10 over the trajectory. Only its
  Julia output is committed, so it is recorded here as an unarchived development check and carries no weight.
- The exact reference is computed by two separately developed solvers, QuTiP and QuantumOptics.jl. Their
  burst peaks agree to within 3e-6 (relative) on the six evaluated cells, in the outside collaborator's run.
- None of this is independent replication in the sense of `EXTERNAL_VALIDATION_STATUS.md`: the outside run used
  this repository's model definitions, parameters and disorder draws.

Not established by any of the above:

- The gates cannot establish that the model in §A.1 is what a reader means by "open disordered
  Tavis–Cummings," since the same person wrote this document and the code and a shared conceptual error
  would survive all of them. That question was therefore put to an outside reviewer, who read
  §A.1 and §A.2 on 2026-09-14 and found them "consistent with the standard phenomenological open
  disordered Tavis–Cummings model," having checked the dephasing convention for a sign or factor-of-two
  error specifically. That review covers the **formulation only**, explicitly not the closure results and
  explicitly not quantitative applicability at every coupling; its full scope is recorded in
  `EXTERNAL_VALIDATION_STATUS.md` (not in the public release).
- The Result 23 linearization, which uses `γ⊥` and the inverted-state bright-mode eigenvalues, still
  awaits an independent theory review. The gates above verify the `γ⊥` input to it, not the
  linearization itself.
