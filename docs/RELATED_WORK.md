# Related work (internal technical record)

Rewritten 2026-10-09 after the literature/novelty audit. Full positioning, the closest-five ranking, the
novelty matrix and the reviewer challenges are in
[`LITERATURE_NOVELTY_AUDIT.md`](LITERATURE_NOVELTY_AUDIT.md). The search record is in
[`LITERATURE_SEARCH_LOG.md`](LITERATURE_SEARCH_LOG.md), and sentence-level support in
[`CITATION_AUDIT.md`](CITATION_AUDIT.md). Every key below is in [`../references.bib`](../references.bib).

For each key paper this records:
- what it establishes;
- what we cite it for;
- what we must **not** claim because it exists.

"Read" means the full text or the relevant sections were read, not just the abstract.

## Closest prior work

**egenlauf2026capturing:** Egenlauf, Březinová, Andergassen & Klopotek, MLST 7, 025062 (2026). Read.
- **Establishes:**
  - neural ODEs trained on exact 2RDMs of a closed, quenched 6-site Fermi–Hubbard chain (TD2RDM/BBGKY) reproduce
    the dynamics without explicit three-particle information, but only where two- and three-particle cumulants
    are strongly correlated;
  - memory-dependent kernels are argued to be needed elsewhere;
  - one model is trained per parameter configuration, and extrapolation is in time;
  - trace and positive-semidefiniteness penalty losses are used.
- **We cite it for:** the ML/reduced-dynamics precedent, and the memory argument (§1, §3, §10).
- **We must not claim:**
  - the first learned quantum closure;
  - the first ML approach tied to a missing higher cumulant;
  - the first to identify a need for memory in learned reduced quantum dynamics.
  - Nor may we say it learns κ3, is an open system, or validates our memory branch.

**freter2025dynamical:** Freter, Fowler-Wright, Cuerda, Lovett, Keeling & Törmä, Nanophotonics 14, 5323 (2025).
Read (relevant sections).
- **Establishes:**
  - dynamical superradiance in the dissipative Tavis–Cummings model (local loss and dephasing), solved exactly to
    140 identical emitters from a fully inverted state via permutation and weak U(1) symmetry;
  - mean field is identically zero from the symmetric inverted state;
  - second-order cumulants agree with exact solutions only at early times there and overestimate the oscillation
    amplitude.
- **We cite it for:**
  - permutation-symmetric exact reach (§1, §4);
  - the second-order failure in this model class (§1, §10);
  - mean-field failure (§2);
  - the reach comparison (§10).
- **We must not claim:**
  - that the second-order overshoot or failure for the inverted cavity burst is new;
  - that exact references are impossible above M ≈ 16 at σ = 0.

**henschel2010cavity:** Henschel, Majer, Schmiedmayer & Ritsch, PRA 82, 033810 (2010). Read (cumulant section and
Appendix B).
- **Establishes:** in a cavity–ensemble cumulant expansion truncated at third order, ⟨a†aσᶻ⟩c is the one
  third-order cumulant that "requires more care" and is kept explicitly, because at strong coupling photon-number
  fluctuations change the inversion substantially.
- **We cite it for:** the prior physical importance of our Aaz channel (§1, §7, §10).
- **We must not claim:** that we discovered the importance of the photon-number–inversion correlation. Our claim
  is the quantified split against a+z, the closed-loop ablation, and the prospective confirmation.

**fasser2025benchmarking:** Fasser, Genes, Ritsch & Holzinger, Photonics 12, 996 (2025). Read in full.
- **Establishes:**
  - collective- and individual-spin cumulant expansions to high order against an exact free-space Dicke solution;
  - peak time and power converge with order, and second order is at the few-percent level;
  - long-time populations are poor;
  - "odd orders show unphysical behavior"; the third order is very unreliable; the fourth improves on the second;
  - mean field fails from the inverted state.
- **We cite it for:** the few-percent second order in free space and odd-order unphysicality (§1, §8, §10); the
  mean-field failure (§2).
- **We must not claim:**
  - discovering that cumulant expansions can fail in superradiance;
  - discovering odd-order unphysicality;
  - the first benchmark of cumulant order in superradiance.
  - **Do not cite it for a second-order peak overshoot.** It reports the opposite in its setting (corrected in
    this audit).

**zens2019critical:** Zens, Krimer & Rotter, PRA 100, 013856 (2019). Read (relevant sections).
- **Establishes:** second- and third-order cumulant expansions, including inhomogeneous broadening, for a driven,
  lossy spin ensemble strongly coupled to a cavity (amplitude bistability, finite-N corrections).
- **We cite it for:** conventional higher-order cumulants with disorder in a cavity exist (§1, §8, §10).
- **We must not claim:** that disorder makes higher-order cumulant treatments impossible, or that third-order
  expansions cannot treat inhomogeneous cavity ensembles.

**holzinger2025symmetry:** Holzinger, Rubies-Bigorda, Yelin & Ritsch, PRR 7, 033203 (2025). Read in full (arXiv v2;
see the version caveat in the audit).
- **Establishes:**
  - free-space ordered arrays are rewritten in a collective spin-wave basis;
  - subradiant modes are truncated, and third-order cumulants are applied to the collective operators;
  - complexity drops by about 10²–10³, reaching about 400 emitters;
  - second order overestimates the peak emission by about 5% (chain) and 10–20% (square array) relative to third
    order.
- **We cite it for:** efficient higher-order methods where symmetry permits (§1, §8, §10).
- **We must not claim:**
  - that higher-order correlation approaches are inherently impractical;
  - that the only alternatives are exact simulation or naive truncation;
  - that its method fails under disorder. It does not address frequency disorder, and we did not test it.

## Cumulant and cluster-expansion foundations

| Key | Establishes | Cited for | Must not claim |
|---|---|---|---|
| kubo1962generalized | Generalized cumulant expansion | §1 method origin | — |
| kira2008cluster | Cluster-expansion representation in quantum optics (PRA 78, 022102) | §1 method origin | — (Fasser et al.'s reference list pairs this title with the wrong DOI; we cite the verified PRA 78 paper) |
| sanchezbarquilla2020cumulant | Cumulant expansion for light–matter interactions in arbitrary structures, truncated at 1st–3rd order | §1 method | — |
| plankensteiner2022quantumcumulants | QuantumCumulants.jl (symbolic cumulant equations for open systems) | §1, §4 (Julia reproduction) | a new cumulant framework |
| fowlerwright2023determining | Cumulant convergence with order may be non-uniform across even and odd orders; error not monotone in N (central spin models) | §1, §8 | a general validity criterion (we tested one and rejected it, Results 22–23) |
| kerber2025cumulant (preprint) | No general applicability/convergence criterion for higher-order cumulant expansions is known; smooth convergence in one model, non-physical solutions in another | §1 | — (re-check publication status at submission) |

## Superradiance, the model, and disorder

| Key | Establishes | Cited for | Must not claim |
|---|---|---|---|
| dicke1954coherence | Superradiance | §1 | — |
| tavis1968exact | The N-molecule–radiation-field (Tavis–Cummings) Hamiltonian | §2 | — |
| gross1982superradiance | Review of collective spontaneous emission | §1 | — |
| stitely2023quantum | Superradiant pulses in the open Tavis–Cummings model build up strong, non-Gaussian correlations | §1 | that correlations matter only in our model |
| lindblad1976generators, gorini1976completely | Lindblad/GKS generators | §1, §2 | — |
| debnath2019collective | Second-order mean field for inhomogeneously broadened emitters in a narrow-linewidth cavity (synchronization, collective Rabi oscillations) | §1 | that second-order cumulants had not been applied to disordered cavity ensembles |
| bychek2021superradiant | Second-order cumulant superradiant lasing with inhomogeneous broadening and varying coupling | §1 | coupling-profile effects |
| zeb2022analytical | Analytical bright/dark-sector solution and Fano resonances of the disordered Tavis–Cummings model (spectroscopy) | §1 | a new analytical solution |
| zhang2025robust | The superradiant burst survives strong spatial and spectral disorder in waveguide QED (semiclassical, large N) | §1 | that disorder controls the burst (our extension was a null, Result 26) |
| hanamura2026inhomogeneous (preprint) | Coupling inhomogeneity as a resource for quantum memories (linearized bright/dark picture) | **supplement only** (Result 23 context) | coupling-inhomogeneity-as-resource |

## Learned closures and learned dynamics

| Key | Establishes | Cited for | Must not claim |
|---|---|---|---|
| duraisamy2019turbulence | Data-driven turbulence closures (review) | §1 paradigm | that learning a closure term is a new concept |
| huang2022machine | Learned moment closure for radiative transfer (learns the gradient of the highest moment), with accuracy and generalizability across benchmark problems | §1 paradigm | the old record's "generalizing across size" and "structure-preserving" (not in Part I; corrected) |
| burles2026machine | Review of ML moment closures for plasma, including the need to test closures inside the evolving system | §1 paradigm | — |
| ma2019model | Mori–Zwanzig-motivated learned memory terms for reduced models | §1, §3 | that our traces are a Mori–Zwanzig derivation |
| lagnese2026neural (preprint) | Neural quantum states for super/subradiant dynamics of ~40 atoms in free-space arrays | §1, §10 | any cost or accuracy comparison |

## Reference methods and software (Methods only)

| Key | Used for |
|---|---|
| lambert2026qutip | QuTiP 5 (`mesolve`, `mcsolve`; v5.3.0 used) |
| shammah2018open | PIQS (`qutip.piqs`), permutation-invariant open-system solver |
| dalibard1992wave, daley2014quantum | Quantum trajectories |
| verstraete2004matrix, zwolak2004mixed | MPS for open (mixed-state) dynamics |
| hauschild2018tenpy | TeNPy, used by the MPS baseline |
| kramer2018quantumoptics | QuantumOptics.jl, used by the Julia reproduction |

## Removed

- **verstraelen2023quantum** (truncated-cumulant trajectories for open spin lattices): uncited after the
  restructure and not needed for any claim. It was deleted from `references.bib`.
