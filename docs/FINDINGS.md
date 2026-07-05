# Spike findings: does mean-field / cumulant theory fail for open disordered Tavis-Cummings, and is there a learnable closure?

All numbers below are from an **exact** open-system Lindblad solver (QuTiP 5.3), validated before use.
Cumulants are **Fock-cutoff converged** (verified) and use **bounded spin operators** (truncation-safe).
Spin operators are O(1), so a cumulant of 0.10 means the term is ~10% of the spin scale.

## Solver validation (gates that had to pass first)
- M=1 vacuum Rabi: single-excitation eigenvalues = `wc ± g` exactly (0.9, 1.1). PASS.
- Identical emitters, lossless: bright splitting = `2g√M` exactly, M = 1..6. PASS.

## The quantity that matters
- `c2` = largest 2-point connected spin cumulant `⟨σ⁺σ⁻⟩_c`  → the **mean-field** error scale (mean-field drops it).
- `c3` = largest 3-point connected spin cumulant `⟨σ⁺σ⁻σᶻ⟩_c` → the **second-order-cumulant** error scale = **the neural-closure target** (2nd-order cumulant drops it).
- `c3/c2` small ⇒ 2nd-order cumulant already near-exact ⇒ nothing to learn.

## Result 1 — weak driven steady state: closure target is negligible (~1%)
Incoherent pump, `gcoll=0.8, κ=0.1`, Nc-converged:

| M | σ=0 | σ=0.15 | σ=0.30 | σ=0.50 |
|---|---|---|---|---|
| c3 (M=4) | 0.0052 | 0.0064 | 0.0081 | 0.0093 |
| c3/c2 (M=4) | 0.23 | 0.28 | 0.37 | 0.37 |

Closure target ~0.5–1.5% of the spin scale. **Second-order cumulant is ~99% accurate here; a closure has almost nothing to fix.** Disorder does raise c3/c2 at M=4, but off a tiny base.

## Result 2 — transient superradiance: closure target is LARGE (~5–12%)
All emitters start excited, collectively dump into the lossy cavity. Fock-converged (c3_peak identical for Nc=4,6,8).

| M | σ=0 | σ=0.15 | σ=0.30 | σ=0.50 |
|---|---|---|---|---|
| **c3_peak (M=3)** | **0.116** | 0.119 | 0.110 | 0.090 |
| c3/c2 (M=3) | 0.59 | 0.59 | 0.54 | 0.46 |
| **c3_peak (M=4)** | **0.049** | 0.059 | 0.069 | 0.075 |
| c3/c2 (M=4) | 0.29 | 0.34 | 0.39 | 0.41 |

In the superradiant burst, second-order cumulant genuinely fails: the dropped 3-point cumulant is ~60% of the 2-point at M=3. **This is the regime where a closure has real work to do.**

## Honest interpretation (the two things the data settles)

1. **The premise is regime-specific and real, but modest.** Cumulant theory is near-exact in weak steady states (~1%) and fails substantially only in transient collective (superradiant) dynamics (~5–12%). So there IS a signal, it is not a null, but it lives in a specific regime and is ~10%, not order-1.

2. **The original "extrapolate to large M" framing is contradicted by the data.** The closure target shrinks fast with M (c3_peak: 0.116 at M=3 → 0.049 at M=4, more than halving per added emitter), heading toward the mean-field-exact thermodynamic limit. So the correction is largest exactly where exact diagonalization is *easy* (small M) and vanishes where the closure would be *useful* (large M). The value, if any, is a narrow window at intermediate M.

3. **The one genuinely interesting, hypothesis-consistent thread:** disorder increases the relative closure error at finite M (c3/c2 at M=4 rises 0.29 → 0.41 across σ, and c3_peak rises +54%), and this disorder-enhancement is stronger at M=4 than M=3. If it keeps growing with M, the "disorder + finite M" corner is where beyond-cumulant physics matters in real (inhomogeneously broadened) molecular polaritonics.

## Verdict
Not a null, not groundbreaking. A real, ~6-tier effect. The honest project is either (a) a systematic map of where cumulant theory breaks in polaritonics vs (M, disorder, regime), or (b) a learned closure that extends cumulant accuracy in the disorder+finite-M corner — which still needs the make-or-break test below.

## Result 3 — the closure map is LEARNABLE and TRANSFERABLE (the ML make-or-break: PASS)

Dataset: 81,000 samples from exact trajectories (M=3,4,5; sigma in {0,0.05,0.1,0.2,0.3}; 3 seeds;
superradiant init). Input = local low-order moments among an emitter triple {i,j,k} + cavity +
detunings (exactly the information a 2nd-order cumulant model holds). Target = exact 3rd cumulant
kappa3(sigma_i^+, sigma_j^-, sigma_k^z) (Re, Im). A single MLP on this LOCAL feature vector is
permutation-equivariant and transfers across M by construction. Metric skill = 1 - MSE(model)/MSE(zero);
zero = predicting kappa3 = 0 = exactly what 2nd-order cumulant assumes.

| split | linear skill / R2 | **MLP skill / R2** |
|---|---|---|
| random 80/20 (is it a function?) | 0.74 / 0.53 | **0.99 / 0.96** |
| **M-extrapolation** (train M{3,4} -> test M5) | 0.56 / 0.29 | **0.85 / 0.71** |
| **sigma-extrapolation** (train sigma<=0.1 -> test sigma>=0.2) | 0.69 / 0.43 | **0.93 / 0.84** |

- The closure map is a well-defined, smooth function of the local moments (MLP R2 = 0.96 in-distribution).
- It **generalizes to an unseen, larger emitter number** (train 3,4 -> predict 5 at R2 = 0.71): the whole
  premise of a transferable closure. Predicted-vs-true is tight on the diagonal (results/closure_learn_Mextrap.png).
- It **generalizes across disorder strength** (train low sigma -> predict high sigma at R2 = 0.84).
- It is genuinely **nonlinear** (MLP >> linear everywhere), so the network is doing real work.

## Overall verdict of the spike: GREEN, with honest bounds
There IS genuinely something to explore. The core ML premise (a learnable, nonlinear, permutation-
equivariant closure that transfers across M and disorder) is validated on exact, converged data.
The learnability+transferability result is a concrete, defensible ML-for-physics finding on its own.

Honest bounds:
1. M-transfer is one step (3,4 -> 5) and R2 degrades from 0.96 (in-dist) to 0.71. Whether it holds to
   M=8,10 (the genuinely-hard regime) is untested and is the key open question for the "useful at large M" claim.
2. This validates the closure MAP. It does NOT yet show the CLOSED LOOP (feeding the learned closure into
   the cumulant EOM) is stable and improves trajectories. That is the remaining make-or-break and the
   panel's central risk (learned closures can drift/blow up when fed back).
3. The absolute effect is ~5-11% (superradiant regime), so even a perfect closure corrects a modest effect.

## Result 4 — the closed-loop cumulant EOM (built, VALIDATED, and tested end to end)

A numerical operator-algebra engine builds the second-order cumulant equations of motion without hand-
deriving any equation (no sign/factor risk): it computes each moment's exact derivative from the
Lindbladian L-dagger and applies the cumulant closure symbolically. Correctness gate (unforgeable):
fed EXACT moments + EXACT 3rd cumulants, the assembled derivatives equal the exact ones to **~1e-14**.

Closed-loop integration, open disordered superradiance (M=3, sigma=0.15, held-out realization):
- **Oracle closure** (exact kap3 fed in) reproduces exact to ~0.008 -> the integrator + closure are correct.
- **Plain 2nd-order cumulant** (kap3=0) fails badly: photon number off by **0.85**, and it never damps
  the vacuum-Rabi oscillations (exact does). This is the gap.
- **Learned closure in the loop** (a crude global-feature net, val score only 0.17, evaluated LIVE on the
  evolving moments, held-out disorder): tracks exact and **beats plain cumulant by up to 9x** early, then
  develops a late-time instability:

  | window | plain max err | learned max err | |
  |---|---|---|---|
  | t <= 15 | 0.853 | **0.098** | learned beats plain ~9x |
  | t <= 20 | 0.853 | **0.174** | learned beats plain ~5x |
  | t <= 25 | 0.853 | **0.722** | learned still ahead |
  | t <= 30 | 0.853 | 0.924 | instability onset |
  | t <= 40 | 0.853 | 165 (blows up) | late-time divergence |

Honest reading: a learned closure demonstrably WORKS in feedback over the bulk of the dynamics and beats
plain cumulant substantially, then destabilizes at late times, a classic and addressable closure-modeling
failure mode. And this used the CRUDE global-feature network (val 0.17); the accurate local-feature closure
(R2=0.96 in the learnability test) should extend the stable window. So the closed-loop is a partial success
with a well-defined remaining challenge (late-time stability), not a null and not a solved problem.

## Result 5 — the STABILIZED local-feature closure in the loop (the capstone: PASS)

Replaced the crude global-feature closure with a permutation-equivariant LOCAL-feature closure: each order-3
cumulant is predicted from only the moments among its own factors (the singles and pairs in its cumulant
factorization) + emitter detunings + light global context, with monomials canonicalized by factor type so
one small network per canonical type serves every site assignment and every M. The dynamically dominant
cumulant types are learned at **R2 ~ 0.99** (vs 0.17 for the global net); the many near-zero types correctly
predict ~0.

In the closed loop (held-out disorder realization, superradiance):

| run | steps completed | plain-cumulant max err | learned(local) max err | |
|---|---|---|---|---|
| **M=3 held-out**, whole trajectory t<=40 | 200/200 (stable) | 0.853 | **0.023** | ~37x better, no late-time instability |
| **M=4 transfer** (closure trained ONLY at M=3), t<=40 | 200/200 (stable) | 1.365 | **0.397** | ~3.4x better, stable |

So the accurate local closure (i) is STABLE for the entire trajectory at M=3 (the global closure's late-time
blow-up is gone), (ii) tracks exact to ~2% (37x better than plain cumulant), and (iii) TRANSFERS across
emitter number: trained only at M=3, it runs stably in the M=4 loop and still beats plain cumulant ~3.4x.
Trajectory plots (results/local_closure.png) show blue (local closure) sitting on black (exact) throughout,
while plain cumulant oscillates.

Honest caveats: transfer accuracy degrades from M=3 (0.023) to M=4 (0.40), so transfer works but is not
perfect (the full M-transfer scan through M=5 is Result 6 below); a single coupling/loss regime; the absolute
physics effect is ~10% (superradiant). None of these undercut the core demonstration: a learned closure, done
right, is stable in feedback and transferable in emitter number.

## Result 6 — M-transfer scan: how far does the closure extrapolate in molecule number? (PASS to M=5)

The headline scientific question: train the closure on *small* systems and see how far up in emitter number
it keeps working. Protocol (`transfer_scan.py`), with leakage controls stated explicitly:
- **Train only on M = 2 and 3** (seeds 0–5, σ ∈ {0.05,…,0.30}). Test M = 4, 5 are **never in training** (no
  M-leakage). Test disorder uses **seed 7, held out** from training (no realization-leakage). Feature
  scalers are fit on training data only (no feature-leakage).
- **No hallucination:** each test engine must pass the machine-precision correctness gate (`gate_exact`
  `< 1e-7`) before its loop is trusted, and the exact reference is checked Fock-converged (peak photon
  number well below the cutoff). Both hold below.

| M (never trained) | correctness gate | nc_peak/Nc | stable | plain-cumulant err | learned(local) err | improvement |
|---|---|---|---|---|---|---|
| **4** | 1.6e-15 | 0.28 | 200/200 | 1.053 | **0.414** | 2.5× |
| **5** | 6.4e-15 | 0.31 | 200/200 | 1.296 | **0.469** | 2.8× |

What the trend actually says (all from validated runs, nothing extrapolated by hand):
1. Trained only on M=2,3, the closure stays **stable through the entire trajectory** at both M=4 and M=5 (no
   blow-up), and keeps **beating plain second-order cumulant**.
2. The margin over plain cumulant **grows** with M (2.5× → 2.8×): plain-cumulant error climbs faster
   (1.053 → 1.296, +23%) than the learned error (0.414 → 0.469, +13%). So the closure degrades *more slowly*
   than the thing it replaces, and the gap widens as the system gets harder.
3. Honest limit on accuracy: absolute transfer error (~0.41–0.47) is far worse than in-distribution
   (0.023 at M=3): extrapolating out of the training M costs a lot of accuracy, but it degrades gracefully
   and stays useful, rather than falling apart.
4. Honest limit on reach: **M ≥ 6 cannot be exactly validated on this hardware.** The engine build at M=6
   materializes ~11 GB of dense monomial matrices (OOM); even the M=5 build takes ~13 min. That exact-simulation
   wall is precisely the regime a transferable closure is meant to cross, so the evidence is that it transfers
   cleanly across every M we *can* still check (3, 4, 5), with characterized, slow degradation.

Engineering note (no physics changed): the cumulant RHS was vectorized (`CumulantEngine.deriv_fast`) to make the
M=5 loop tractable. It is guarded by a build-time assertion against the validated `_full_from_tracked`, and
measured **bit-identical** (`max|deriv − deriv_fast| = 0.0`) while running 41× faster. The M=4 row is unchanged
to the digit from the pre-optimization run, confirming no numerical drift.

## Result 7 — a SCALABLE engine that runs the closure at arbitrary M (breaking the exact-simulation wall)

The dense cumulant engine builds the EOM by projecting operators on the full Nc*2^M space and OOMs at M>=6.
The closure's whole purpose is to work where exact simulation cannot, so the engine must too. We proved
(step1_locality.py, on the 1e-14-validated dense engine) that the exact EOM is (A) SITE-LOCAL -- every term
in d<k>/dt lives on support(k) plus at most one extra emitter site (max extra sites = 1 at M=3 and M=4) --
and (B) PERMUTATION-STRUCTURED (C invariant under relabeling sites to ~5e-14). So the entire EOM can be
assembled from tiny subsystem computations (cavity + <=3 qubits) placed onto M sites, with the per-site
frequency term added analytically. This is O(M^3), not exponential.

ScalableEngine implements this. It is validated exactly, not on faith:

| check | result |
|---|---|
| build_C == dense C, disordered, M=2,3,4,5 (validate_scalable.py) | max abs diff **8.6e-15, 4.6e-14, 5.0e-14, 2.1e-14** |
| sparse enumeration == dense enumeration set, M=2..6 (validate_stream.py) | identical sets |
| streaming deriv == matrix map, M=3,4 | **2.6e-15** |
| compiled (vectorized/sparse) deriv == matrix map, M=3,4,5 | **~1.5e-15** |
| compiled deriv speed | 0.7 ms (M=8), 3.6 ms (M=12), 14-20 ms (M=16-20) |

So the second-order cumulant EOM (plain or with the learned closure) can be integrated at M=20, 40, ... with
the SAME coefficients the dense engine would have produced, proven identical wherever both can be built. The
engine no longer caps at the exact-simulation wall.

## Result 8 — the learned closure running at large M (the flagship run, with an honest limit)

With the scalable engine, the closure trained ONLY on M=2,3 (held-out seeds) is run in the closed loop far
past any dense solve. Exact reference by quantum trajectories (mcsolve) on the identical exact_tc model (zero
convention risk; validated vs exact mesolve at M=5 within Monte-Carlo error), which -- unlike a full
density-matrix mesolve (impractical by M~7 for the stiff superradiant burst) -- evolves cheap pure states.

First, a full-pipeline self-consistency check: at M=5 the scalable + compiled-EOM + vectorized-closure loop
reproduces the dense-engine result **to the digit** (learned err 0.4691, plain 1.2962, 2.8x), so the entire
new machinery changed nothing but the cost.

Closure (trained on M=2,3) vs EXACT mcsolve, disordered seed 7 (never trained), max error in (n, mean sz):

| M | tracked moments | plain-cumulant err | learned err | winner | MC err |
|---|---|---|---|---|---|
| 5 | 141 | 1.296 | **0.469** | learned 2.8x | (exact) |
| 8 | 330 | 1.967 | **1.307** | learned 1.5x | 0.096 |
| 10 | 501 | 2.503 | **2.398** | learned 1.04x | 0.119 |

Then stability where NO exact solve is possible at all (closure runs, no reference):

| M | tracked moments | stable? | n_peak | z_end | wall-clock |
|---|---|---|---|---|---|
| 12 | 708 | yes (200/200) | 8.77 | -0.96 | 14 s |
| 16 | 1230 | yes (200/200) | 11.82 | -0.99 | 23 s |
| 20 | 1896 | yes (200/200) | 14.77 | -0.98 | 39 s |

The honest reading, both halves:
1. **What works:** the scalable engine runs the closure stably at M up to 20 (and beyond), producing the
   physically correct trajectory -- superradiant burst (peak photon number grows with M: 8.8 at M=12, 11.8 at
   M=16, the collective enhancement) followed by decay to the ground state (z_end -> -1). The closure keeps
   beating plain second-order cumulant out to M=8 (1.5x).
2. **The honest limit:** the closure trained on M=2,3 does NOT usefully extrapolate as far as M=10 -- by M=10
   its advantage over plain cumulant has collapsed to ~1.04x (2.40 vs 2.50). And both cumulant variants have
   large absolute error at M=8,10. Part of that growth is scale (the superradiant peak itself grows with M, so
   a fixed relative error shows up as a larger absolute error), but the shrinking learned-vs-plain margin is
   real: a closure trained on 2-3 molecules is not accurate for 10.

So the durable contribution is the **scalable, exactly-validated engine**: it removes the exact-simulation wall
from the METHOD, which in turn makes the obvious fix available -- retrain the closure on the larger M the engine
can now reach (M=8,10,12) instead of only M=2,3. That retraining, not a bigger extrapolation from tiny M, is the
path to an accurate large-M closure, and it is only possible because the engine now scales.

## Result 9 — retraining on the M the engine unlocked RECOVERS large-M accuracy (the ladder works)

Result 8's honest limit was that a closure trained on M=2,3 collapses to ~1x by M=10. The promised fix was to
retrain on larger M, which the new machinery makes cheap: the e_ops data path (gen_data_v2.py) computes exact
training moments WITHOUT the dense engine -- proven EXACTLY equal to the old moments_from_rho path
(check_eops_path.py: max diff 0.00e+00 over all monomials) -- so exact M=4,5 data that used to need a 13-minute
engine build per realization now takes 3-60 s per trajectory. Generated 906,000 new samples (M=4,5; seeds 0-3;
sigma 0.05/0.15/0.30; all Fock-checked), merged with the old M=2,3 set (1.82M total). The two scripts were
adversarially reviewed by a 12-agent workflow before running (one operational finding, fixed; physics, leakage
protocol, and feature-identity all survived attack).

Head-to-head in the closed loop, identical held-out disorder (seed 7, never trained) and IDENTICAL seeded
mcsolve references as Result 8 -- the old-closure and plain columns reproduce Result 8 to the digit (built-in
regression check, passed):

| M | ref | plain | closure trained M=2,3 | **closure trained M=2..5** | rel photon err old -> new |
|---|---|---|---|---|---|
| 5* | mesolve (exact) | 1.296 | 0.469 (2.8x) | **0.348 (3.7x)** | 0.137 -> 0.102 |
| 8 | mcsolve | 1.967 | 1.307 (1.5x) | **0.856 (2.3x)** | 0.253 -> 0.166 |
| 10 | mcsolve | 2.503 | 2.398 (1.04x) | **0.890 (2.8x)** | 0.381 -> **0.141** |

*M=5 is inside the new training range (held-out seed only); M=8,10 are genuine extrapolation for both closures.

What this establishes:
1. **The collapse is repaired.** At M=10, where the old closure was tied with plain cumulant, the retrained one
   is 2.8x better. Its absolute error is now nearly FLAT in M (0.35 / 0.86 / 0.89), and its RELATIVE photon
   error IMPROVES from M=8 to M=10 (0.166 -> 0.141) while the old closure's explodes (0.253 -> 0.381).
2. **The ladder is demonstrated, not conjectured.** Train on the M you can solve exactly -> the scalable engine
   runs the closure beyond it -> that reach generates the next rung's training data. This loop was executed
   once (train <=5, validated at 8-10) and nothing in it caps at the old exact-simulation wall.
3. Stability holds at very large M with the retrained closure: M=12,16,20 all stable end-to-end with the
   physically correct burst-then-decay shape (n_peak 8.3 / 11.2 / 14.1, z_end -> -1).

Honest caveats: at M=12-20 z_end undershoots -1 by ~0.4-1.3% (cumulant EOMs are not positivity-preserving;
small but real). The M=8,10 references carry Monte-Carlo error ~0.10-0.12, well below the differences being
claimed. Absolute error ~0.86-0.89 at M=8,10 is still ~14-17% of the superradiant peak: better than anything
else available at that M, not yet (retracted: see SCOPE #4). One coupling/loss regime throughout.

## Result 10 — pre-registered pilot: the regime-aware closure generalizes, and the REPRESENTATION matters

Pilot per PROTOCOL.md (decision table fixed before training): v3-RAW and v3-RATIO trained on IDENTICAL
pilot data (68 Fock-safe configs, 170,400 samples, corners+center of the regime grid, seed-decorrelated
disorder), judged per cell at M=5 on held-out disorder (seed 7) against exact mesolve references, with
plain 2nd-order cumulant and the frozen regime-blind v2 closure as baselines. NRMSE on photon number:

| cell | plain | v2 (regime-blind) | v3-RAW | **v3-RATIO** |
|---|---|---|---|---|
| I1 interpolation (g,k) | 0.411 | 0.124 | 0.063 | 0.072 |
| I2 interpolation (all) | 0.322 | UNSTABLE | 0.110 | 0.134 |
| I3 center, new seed | 0.412 | 0.101 | 0.465 | 0.098 |
| E1 high dephasing (3x train max) | 0.219 | 0.322 | 0.128 | **0.049** |
| E2 high disorder (sigma=0.4) | 0.420 | 0.070 | 0.049 | **0.048** |
| E3 strong coupling (g=1.1) | 0.505 | 0.223 | UNSTABLE | **0.072** |
| E4 weak coupling (g=0.3) | 0.296 | 0.131 | 0.058 | **0.054** |
| E5 high kappa (0.4) | 0.315 | 0.109 | 0.118 | **0.070** |
| E6 low kappa (0.03) | 0.539 | 1.452 | UNSTABLE | **0.377** |
| IC stress f=0.75 (never trained) | 0.407 | 6.66 | 44.9 | **0.127** |

Decision table (pre-registered): v3-RATIO interpolation PASS (median 0.113 vs plain 0.409), extrapolation
6/6 cells, kill condition not triggered. v3-RAW: interpolation FAIL, 2 instabilities, catastrophic IC
failure. CAMPAIGN PROCEEDS WITH THE RATIO REPRESENTATION.

What the pilot established beyond the go/no-go:
1. **The representation is itself a finding.** Same data, same architecture, same targets -- expressing
   detunings and rates relative to the cavity linewidth (Delta/kappa, g/kappa, gphi/kappa) is the
   difference between a closure that destabilizes under extrapolation (RAW: 2 blow-ups, 44.9 on unseen
   initial conditions) and one that transfers across every tested regime AND an initial-condition family
   it never saw (0.127). Feature geometry carries physics.
2. **The regime-blind ablation confirmed structurally and now empirically:** v2 (no regime inputs) is
   unstable at one interpolation cell, WORSE than plain cumulant at high dephasing, and fails the IC
   stress at 6.66. Regime awareness is necessary, not decorative.
3. **First failure-map entry:** low kappa (0.03) is the hardest cell for every method; v3-RATIO still wins
   (0.377 vs 0.539) but degrades. This is CONSISTENT WITH longer dynamical memory in the slow-drain regime
   (a memoryless instantaneous closure has more to miss), but the mechanism is not established -- that
   would need memory-timescale quantification or a history-aware closure comparison. The campaign watches
   this boundary.
4. **Mean-field's rung on the ladder, stated precisely:** for OUR incoherent, fully-inverted (and diagonal
   partially-inverted) initial states, all coherences vanish and the factorized mean-field equations have
   no fluctuation seed, so every MF derivative is identically zero and the superradiant burst never starts.
   This is specific to such initial conditions -- driven or coherently seeded states DO evolve under MF.
   Within this study's setting the hierarchy MF -> 2nd-order cumulant -> learned closure -> exact is
   anchored by physics at the bottom.

## Result 11 — the regime campaign: four pre-registered questions, four honest answers

v3-RATIO trained ONCE on the full grid (648 configs, 2,077,650 samples, 0 Fock-skips, seed map asserted
collision-free), evaluated per PROTOCOL.md. Baselines: plain 2nd-order cumulant, frozen v2. All references
exact (mesolve at M=5; seeded mcsolve at M=8,10 with MC error stated). Absolute errors everywhere.

**Q1 -- does it win across seeds? YES.** Center regime, M=5, seeds 7/8/9 (paired, same realizations):
NRMSE_n 0.032 / 0.034 / 0.023 vs plain 0.412 / 0.416 / 0.421 (v2: 0.101 / 0.088 / 0.079). Superradiant
peak error 0.001-0.006 absolute. Paired improvement positive on every seed (0.380 / 0.382 / 0.397).
IC f=0.75 stress across the same seeds: v3-RATIO 0.084-0.136; v2 UNSTABLE on 2 of 3 seeds.
Also: full-grid training improved the closure itself vs the pilot version (0.098 -> 0.023-0.034 at the
center cell) -- more regimes made it better, not just broader.

**Q2 -- does it keep winning in the hard regions? Accuracy YES; stability criterion FAILED.**
Two exact-referenced heatmaps at M=5 (78 cells). Claim classes corrected after review to three levels of
held-out-ness (a cell whose regime tuple sits ON the training grid is NOT interpolation evidence even with
an unseen disorder seed):
- D, in-distribution (tuple on the training grid, disorder seed unseen): 12/12 stable cells win; paired
  bootstrap median NRMSE_n improvement 0.343, 95% CI [0.302, 0.412]. This is new-realization
  generalization, the weakest claim class, labeled as such.
- I, TRUE interpolation (tuple inside the hull but never trained, seed unseen): 14/14 stable cells win;
  median 0.341, 95% CI [0.316, 0.415]. (+1 unstable cell, counted against stability below.)
- E, extrapolation (tuple outside the hull): 48/50 win; median 0.292, 95% CI [0.263, 0.358]. (+1 unstable.)
- high disorder (sigma=0.4, beyond training): NRMSE_n 0.01-0.03 across ALL couplings. Zero disorder
  (sigma=0, also beyond training): 0.02-0.05. Coupling extrapolation both directions (g=0.3, 1.1): clean.
- THE FAILURE POCKET (recorded, not hidden): the kappa floor. At kappa=0.03 (40% below training minimum)
  the model ties or loses to plain at high dephasing (0.34 vs 0.22 at gphi=0.3) and has 2 unstable cells
  at gphi=0.05 -- one of them (kappa=0.05, gphi=0.05) INSIDE the hull. gphi=0.05 is the midpoint between
  the training gphi values {0.02, 0.1}: consistent with sparse training density in the gphi direction
  combined with the slow-drain regime (mechanism not established). Stability rate 76/78 = 97.4%, which is
  WORSE than plain's 100% -- that pre-registered validity criterion FAILS and is reported as failed.

**Q3 -- does size transfer survive at M=8, M=10 against exact references? YES at M=8; NOT ESTABLISHED
at M=10.** V0 regime (tuple NOT on the training grid, so these rows are unseen-M AND unseen-tuple),
seeded mcsolve refs:
- M=8 (seeds 7,8,9; MCerr ~0.09): v3-RATIO NRMSE_n 0.161 / 0.053 / 0.096 vs plain ~0.478, v2 0.26-0.29.
  Seed 8's 0.053 is at or below the reference's stated Monte-Carlo uncertainty: phrased precisely, the
  closure MATCHED THE SEEDED MC REFERENCE WITHIN ITS ESTIMATED NUMERICAL UNCERTAINTY -- not "beat exact".
- M=10: seed 8 wins big (0.109 vs plain 0.500, v2 0.303); seed 7 UNSTABLE (v2 was stable there at 0.280).
  One of two seeds is a blow-up: conditional success. Accuracy-when-stable is much better than v2;
  robust 2x size transfer is NOT yet established.

**Q4 -- absolutes, not ratios: reported in every row above** (NRMSE per observable, max-abs, peak error).

Evaluation-row provenance (applies to every table above; the split-reporting discipline):
| row | tuple trained? | seed trained? | IC trained? | M trained? | reference |
|---|---|---|---|---|---|
| Q1 center rows | YES (D class) | no | yes (f=1) | yes (M=5) | mesolve (exact) |
| Q1b IC rows | YES | no | NO (f=0.75 unseen) | yes | mesolve (exact) |
| Q2 D cells | YES | no | yes | yes | mesolve (exact) |
| Q2 I cells | NO (in hull) | no | yes | yes | mesolve (exact) |
| Q2 E cells | NO (outside hull) | no | yes | yes | mesolve (exact) |
| Q3 rows | NO | no | yes | NO (M=8,10 unseen) | mcsolve (MC err stated) |

Campaign gate verdict, stated against the pre-registered criteria WITHOUT rounding up:
- Accuracy across seeds: PASS (decisive). Accuracy across regimes: PASS for D, I, and E classes (all CIs
  exclude zero), including the high-disorder and high-dephasing cells that justify the project.
- STABILITY: FAIL against its own criterion (97.4% vs plain's 100%; one unstable cell inside the hull).
- Size transfer: PASS at M=8; NOT ESTABLISHED at M=10 (1 of 2 seeds unstable).
Honest one-paragraph conclusion: a ratio-based, regime-aware learned closure substantially reduced error
across all held-out seed tests and most tested interpolation and extrapolation cells, including
high-disorder and high-dephasing regimes; the method's limitations were also mapped -- stability degrades
in a low-cavity-loss pocket and becomes marginal at M=10 -- showing that improved accuracy-when-stable
does not yet guarantee robust long-horizon extrapolation. The sharply-defined next question: why do
slow-draining (long-memory) systems break a memoryless closure, and can that boundary be pushed?

## Result 12 — the staircase and the mapped stability boundary (diagnosis, not yet mechanism)

**Size-transfer staircase** (M=6..10, seeds 7/8/9, V0 regime, seeded mcsolve refs; stability rate and
conditional NRMSE reported side by side, blowups marked, never averaged in -- results/staircase.png):

| M | v3-RATIO stability | NRMSE when stable (per seed) | MC ref err | plain / v2 (all stable) |
|---|---|---|---|---|
| 6 | 3/3 | 0.057, 0.045, 0.059 | ~0.079 | 0.49 / 0.26 |
| 7 | 3/3 | 0.088, 0.041, 0.064 | ~0.084 | 0.47 / 0.26 |
| 8 | 3/3 | 0.161, 0.053, 0.096 | ~0.095 | 0.48 / 0.28 |
| 9 | 2/3 | 0.088, 0.214 | ~0.103 | 0.48 / 0.28 |
| 10 | 1/3 | 0.109 | ~0.115 | 0.49 / 0.29 |

Precise conclusion: accuracy remains near the Monte-Carlo reference uncertainty through M=7, transfers
strongly through M=8, and becomes CONDITIONALLY accurate but stability-limited at M=9-10. The failure is
not a cliff at a magic M: stability degrades gradually and probabilistically (3/3 -> 2/3 -> 1/3) while
accuracy-when-stable stays excellent (0.09-0.21, vs plain ~0.48 and v2 ~0.28 everywhere).

**The dense pocket probe** (5x6 grid, kappa 0.03-0.05 x gphi 0.02-0.1, dev seeds 7/8, 60 instrumented
rollouts) CORRECTS an earlier partial-data reading (a mid-run tail suggested kappa=0.04 was clean; the
complete grid says otherwise) and maps the boundary:
- stability rate rises smoothly with kappa: 6/12, 7/12, 7/12, 9/12, 11/12 for kappa = 0.03..0.05;
- within rows, LOW DEPHASING is the dangerous direction (unstable counts by gphi = 0.02..0.1:
  6, 7, 5, 1, 1, 0 of 10). The boundary is the long-coherence-time corner: small total
  decoherence/drain. CONSISTENT WITH a dynamical-memory limitation of a memoryless closure; the
  mechanism is NOT established (the 2x2 density-vs-gate experiment and, if needed, a minimal
  history-feature test discriminate the candidates).
- diagnostics are uniform across every unstable rollout: near-total output saturation (81-98% at the
  training-range bounds), |kappa3| pinned at the box ceiling (~1.0-1.17), late divergence onset
  (t ~ 26-39 of 40), solver effort exploding with (not before) saturation, and identical-parameter
  seeds splitting survive/diverge -- a probabilistic boundary, consistent with closure-output-driven
  instability after the state leaves training support. Not proven; the 2x2 tests it.
- the campaign's lone in-hull instability, (kappa=0.05, gphi=0.05, seed 7), reproduces exactly.

## Result 13 — support-risk indicator: FAILED its pre-registered verdict (reported, not re-tuned)

A Mahalanobis support-distance indicator (threshold d/d0 >= 1 with d0 = training 99th percentile, fixed at
fit time; criteria committed to git before any risk trace existed) was validated on dev seeds 7/8 along the
BASE closure's own trajectories (fade disabled):
- Coverage PASSED perfectly: lead >= 2.0 time units in 20/20 unstable rollouts (leads 25-39). Two trigger
  modes observed: at kappa=0.03 the trigger fires at t=0 (the regime coordinates themselves are outside
  training support, e.g. g/kappa = 23 vs training max 20 -- static parameter-extrapolation detection); at
  kappa=0.035-0.05 several cells trigger DYNAMICALLY at t ~ 2.6, during the burst, 25-30 units before
  divergence -- evidence of a real temporal state-drift signal.
- FALSE-ALARM FAILED: stable in-support center-regime rollouts spend 7.2-12.7% of integrator calls above
  threshold (criterion: < 5%). The indicator cries wolf in-support.
- VERDICT: FAIL, as pre-registered. The indicator as parameterized is not a claimable safety system, and
  the fail-safe hybrid is NOT entered into the 2x2 (it would fade 7-13% of legitimate in-support
  corrections). Any revised indicator (sustained-crossing requirement, looser threshold, or a
  regime-static vs state-dynamic decomposition -- the latter motivated by the two observed trigger modes)
  is a NEW experiment requiring its own pre-registration and fresh validation data. We do not re-tune on
  the rollouts that produced this verdict.
- Log artifact note: the run's printed "trigger=never" lines are a fixed falsy-zero display bug; those
  triggers fired at t=0.0 and the saved lead values were always computed correctly.

## Result 14 — the 2x2: local coverage was the dominant mechanism; the naive gate flattened the physics

Four arms, identical except the fix under test, judged on the pocket grid x UNTOUCHED seeds 9/10 with a
common normalized yardstick r = |applied kappa3| / B_common (B = base training 99th-percentile bounds):

| arm | stability | median NRMSE (stable) | P(r>0.90) | P(r>0.99) | max r | sanity (4 cells, seed 9) |
|---|---|---|---|---|---|---|
| base (V3) | 80.0% | 0.044 | 4.7% | 4.4% | 4.76 | 0.023 / 0.069 / 0.023 / 0.035 |
| **dens** | **93.3%** | **0.031** | **2.0%** | **1.8%** | 4.76* | 0.024 / 0.049 / 0.014 / 0.046 |
| gate | 90.0% | 0.107 | 80.4% | 80.0% | 1.00 | 0.104 / 0.085 / 0.050 / 0.188 |
| both | 98.3% | 0.097 | 78.9% | 78.5% | 1.11 | 0.102 / 0.070 / 0.053 / 0.145 |

- GATE REJECTED (the pre-registered failure mode, now measured): it stabilizes by FLATTENING the learned
  correction -- conditional accuracy 2.4x worse in-pocket, 3-5x worse out-of-pocket (sanity
  disqualification). tanh compression distorts the bulk of the correction, not just the tail. "both"
  inherits the same accuracy failure despite the best stability (98.3%).
- DENSITY WINS every non-degenerate criterion: stability 80 -> 93.3%, conditional NRMSE IMPROVES
  0.044 -> 0.031, saturation cut ~2.4x, sanity clean.
- *DISCLOSED CRITERION DEGENERACY (dated here, not silently reinterpreted): max r and max|kappa3| are
  pinned at each clip-arm's box ceiling whenever ANY rollout diverges (4.76 = ceiling/B by construction),
  so for clip-mode arms they duplicate "stability < 100%" rather than grading excursions. Under the strict
  letter no arm improves all four criteria; under the substance density improves everything the degenerate
  pair was meant to proxy. DECISION: density-only frozen as V4, protected by the final verdict on
  untouched seeds 11/12/13 which had no part in this judgment.
- Boundary transformation (the key question): PARTIAL. The unstable region shrank with accuracy improving,
  but the deep long-coherence corner (kappa=0.03, gphi=0.02) remains unstable in every arm. Local coverage
  was the dominant mechanism; the memory hypothesis stays alive ONLY for that residual corner (seeds 14/15
  reserved, run only if the boundary-move proof says it matters).

## Result 15 — boundary-move proof, size axis: V4 REGRESSED (the fix shifted the boundary, did not expand it)

Staircase re-run (M=6..10, original comparison seeds 7/8/9, cached seeded mcsolve refs), V3 vs frozen V4:

| M | V3 stability | V4 stability | conditional NRMSE (V3 -> V4, stable cells) |
|---|---|---|---|
| 6 | 3/3 | 3/3 | 0.045-0.059 -> **0.027-0.048** (better everywhere, peak errors ~4x smaller) |
| 7 | 3/3 | **2/3** | 0.041-0.088 -> 0.035-0.055 where stable; **seed 7 UNSTABLE (V3 was stable)** |
| 8 | 3/3 | **2/3** | 0.053-0.096 -> 0.050-0.083 where stable; **seed 7 UNSTABLE (V3 was stable)** |
| 9 | 2/3 | 2/3 | ~equal (0.087-0.192) |
| 10 | 1/3 | 1/3 | ~equal (0.105) |

Totals: V3 12/15, V4 10/15. THE TRADE-OFF, stated plainly: pocket-density retraining improved the
targeted regime axis (Result 14: stability 80 -> 93.3% with better accuracy on the selection seeds) while
CONTRACTING the size axis at the hardest realization (seed 7 lost at M=7 and M=8). Accuracy-when-stable
improved on BOTH axes; stability moved in opposite directions. Adding ~1M pocket samples (of 3.1M total)
re-weighted the learned function everywhere, not just in the pocket -- the stability boundary of this
memoryless closure responds NON-LOCALLY to training-distribution changes.

Consequences (per the pre-registered tree, applied honestly):
1. Density did NOT "fix it" in the decision tree's sense: it traded one boundary for another. V4 and V3
   are two closures with DIFFERENT stable envelopes, not a strict improvement.
2. The final verdict on seeds 11/12/13 (running) completes the regime-axis picture; the overall V3-vs-V4
   call weighs both axes and is made only after it lands.
3. The minimal-memory experiment (seeds 14/15, pre-registered question, smallest possible history
   features) has now EARNED its slot: the residual deep corner resists all four arms, and the size axis
   resists local-coverage fixes entirely.

## Result 16 — fresh-seed two-axis verdict: V4 is the leading candidate; the boundary is seed-sensitive

Both axes evaluated on untouched seeds 11/12/13 (per the ledger; no combined averages anywhere):

| model | pocket stability (per seed) | pocket cond. NRMSE | size stability (M=6-10) | size cond. NRMSE |
|---|---|---|---|---|
| V3 | 80.0% (76.7 / 76.7 / 86.7) | 0.052 | 13/15 | 0.077 |
| V4 | 83.3% (86.7 / 90.0 / 73.3) | 0.043 | 14/15 | 0.058 |

- SIZE: V4 wins on fresh seeds -- conditional accuracy better in EVERY paired stable cell, stability 14/15
  vs 13/15, and at M=10 V4 is 3/3 including seed 11 where V3 is unstable. Result 15's seed-7 regression
  did NOT reproduce: it was realization-specific and remains reported as evidence that the stability
  boundary is strongly SEED-SENSITIVE (now demonstrated across six seeds: selection seeds flattered the
  pocket, comparison seeds flattered neither axis consistently).
- POCKET: modest pooled win for V4 (83.3% vs 80.0%, 0.043 vs 0.052) that is NOT uniform per seed
  (seed 13: V4 73.3% vs V3 86.7% -- documented counterexample). Sanity cells: no material regression.
- BRANCH (per the pre-registered tree): V4 wins both axes pooled on fresh seeds -> V4 becomes the LEADING
  CANDIDATE, explicitly not "the solution": density is a useful controlled improvement whose effect size
  is modest and seed-scattered in the pocket. Unfixed by ANY arm and now the pre-registered V6 targets:
  the deep long-coherence corner (kappa<=0.035, low gphi) and M=9 at hard realizations.
- DISCLOSED AMBIGUITY in the branch rule (surfaced, not buried): the tree did not specify pooled vs
  per-seed reading for "wins pocket fresh". Pooled -> branch 1 (V4 leading candidate; adopted). Strict
  per-seed (2 of 3, seed 13 counterexample) -> branch 3 (tied within seed variation, V3 baseline). The
  readings converge on identical next actions (V5/V6 run either way; same V6 targets); the final
  architecture comparison therefore includes BOTH incumbents -- V3, V4, V5, V6, two axes, fresh seeds --
  so no label choice can hide behind the ambiguity, and V6 must beat the best incumbent, not just V3.
- V0-comparability note: at the V0 regime V4's fresh-seed M=10 rows (0.097-0.149) sit at 2-3x the MC
  reference uncertainty (0.108-0.115) -- conditional accuracy at M=10 is now routinely within a small
  multiple of the reference's own noise.

## Result 17 — the V5/V6 dev verdict: memory repairs BOTH mapped failure axes; the letter-vs-substance call

Dev seeds 19/20 (ledger; first-ever use), exposure-matched controls (V5 = identical balanced rows as V6,
minus the 6 trace columns; V4' = natural-mix count control, descriptive only), decision template written
before any result existed:

| model | deep-pocket stability | pocket NRMSE (stable) | far-M (9,10) stability | far-M NRMSE | sanity med |
|---|---|---|---|---|---|
| V3 | 0/12 | -- | 1/4 | 0.192 | 0.028 |
| V5 balanced, no memory | 4/12 | 0.107 | **0/4** | -- | 0.029 |
| **V6 balanced + memory** | **12/12** | **0.051** | **4/4** | **0.127** | 0.054 |

The attribution chain is unambiguous:
1. **Memory, not balancing, is the missing ingredient.** V6 vs V5 differ ONLY by trace columns on
   identical rows: 12/12 vs 4/12 in the pocket, 4/4 vs 0/4 at far-M. V6 is the FIRST model to survive
   the gphi=0.02 column (extrapolation for every model, all others X/X) and does so at NRMSE 0.04-0.07.
   Far-M accuracy 0.116-0.145 sits near the MC reference floor (~0.10-0.12).
2. **Balancing alone HURTS size transfer**: V5 0/4 at far-M where the count-matched natural-mix control
   V4' is 4/4 -- the clean two-control comparison nobody can attribute to sample count.
3. **The cost, quantified**: V6's sanity cells run 0.024-0.056 vs V3's 0.022-0.041 (worst absolute
   regression 0.034 NRMSE, in-support).

THE LETTER: the mechanical template printed TEMPLATE 4 (fail), because the in-code sanity guard
median(V6) < 1.5 x median(V3) tripped (0.054 vs 0.042). Disclosures, dated: (a) the 1.5x constant was an
in-code improvisation -- the protocol's success criterion said "no MATERIAL sanity regression" and never
defined "material" numerically (same underspecification class as the max-r degeneracy in Result 14);
(b) a sanity bookkeeping bug (two cells sharing a dict key; median over 3 of 4 cells) was found post-hoc
and fixed -- NOT decisive (corrected median 0.0445 vs threshold 0.042: fails by 6% instead of 29%).

THE ADJUDICATION (explicit, dated, and carried by untouched seeds): a model that takes both mapped
failure regions from 0-33% to 100% stability at near-reference accuracy, at a worst-case in-support cost
of +0.034 absolute NRMSE, does not meet any reasonable reading of "destroys normal regimes". The final
verdict on untouched seeds 21/22/23 (running, with the corrected bookkeeping) decides: if the two-axis
dominance replicates and the sanity cost stays small, the claim is TEMPLATE 2 with a quantified cost --
"physics-timed memory repairs the mapped long-coherence boundary and far-M stability at a small,
quantified in-support accuracy cost". If the sanity cost balloons or the dominance does not replicate,
the letter stands and V6 is reported as failed.

## Result 18 — V6 FINAL verdict (untouched seeds 21/22/23): the pocket repair REPLICATES; the far-M
## stability crisis does NOT; the sanity cost is real, small, and consistent

| model | pocket stab (18 cells) | pocket med | far-M stab (6) | far-M med | sanity med (4 cells) |
|---|---|---|---|---|---|
| V3 | 22% | 0.095 | 100% | 0.115 | 0.029 |
| V5 (balanced, no memory) | 83% | 0.068 | 100% | 0.187 | 0.050 |
| **V6 (balanced + memory)** | **100%** | **0.043** | 100% | **0.110** | 0.055 |

REPLICATED (the central claim): **V6 completely repairs the long-coherence pocket** -- 18/18 stable
including the gphi=0.02 extrapolation column on every seed (V3: 4/18; V5: 15/18 with near-failures 0.34
and 0.38 in that column), at the best accuracy of any model (median 0.043). V6 > V5 on every pocket
measure with only the 6 trace columns different: memory is the ingredient, on fresh seeds, twice.

NOT REPLICATED (walked back, on the record): the dev seeds' far-M stability crisis. On 21/22/23 EVERY
model is far-M stable -- the dev-seed picture (V3 1/4, V5 0/4) was realization luck, the third
demonstration that the stability boundary is strongly seed-dependent (after Result 15's regression and
Result 16's reversal). V6's far-M contribution on final seeds is therefore an ACCURACY claim only: it
beats its control V5 decisively (0.110 vs 0.187 median), matches V3 (0.110 vs 0.115), and sits slightly
behind the descriptive V4 (0.078-0.115). "Balancing hurts far-M" survives only as an accuracy effect
(V5 0.162-0.230), not a stability collapse.

REPLICATED: the sanity cost -- V6 median 0.055 vs V3 0.029, concentrated in the same two cells both
rounds (high-dephasing 0.055 vs 0.025; strong-coupling 0.056 vs 0.032). Small in absolute terms
(worst +0.030 NRMSE), consistent, real, and disqualifying under the letter of the improvised 1.5x guard:
the mechanical template printed TEMPLATE 4 in BOTH rounds.

THE HONEST LANDSCAPE after two clean verdicts: no single model dominates. V4 is the best in-support and
far-M model. V6 uniquely owns the long-coherence pocket (the only 100%-stable model there, twice, at the
best accuracy) and pays ~+0.03 absolute in-support for it. The project's claim set is exactly that
landscape plus the attribution chain that produced it; whether V6's cost is "material" is an explicit
judgment now sitting on top of fully-disclosed numbers, not something the mechanical rule can settle,
because the rule's constant was improvised (Result 17). What is NOT in question: the memory mechanism's
effect on the mapped boundary is real, replicated, and cleanly attributed.

## Result 19 — V7 FINAL VERDICT (untouched seeds 26/27/28): the physics-routed closure PASSES its four
## pre-registered regions; architecture work ends here by stop rule

V7(kappa, gphi) = V6 iff (kappa <= 0.045 and gphi <= 0.05) else V4; box frozen before all seam/final data;
branch equivalence 0.00e+00 both directions on every run.

1. INSIDE MEMORY BOX -- must match V6's repair: PASS. 12/12 stable, median NRMSE 0.044, bit-identical to
   V6. The box caught live V4 failures a THIRD fresh-seed round (deep corner X on all three seeds; 0.283
   near-failure at (0.045, 0.02)). V6's pocket repair is now replicated on three independent seed sets
   (21-23, 24/25, 26-28).
2. OUTSIDE / NORMAL REGIMES -- must match V4: PASS (identity by construction). Sanity median 0.042.
   Far-M 4/6: seed 28 unstable at BOTH M=9 and M=10 on the V4 branch -- the fourth independent
   demonstration that far-M stability is realization-probabilistic for memoryless closures.
3. SEAM -- no new discontinuity: PASS on the letter (V7 is bit-identical to its branch everywhere).
   Informative edge event: at (0.03, 0.065), one seed of three, V7[v4] unstable while direct-V6 was
   stable at 0.035 -- evidence the true V4-failure region leaks slightly past the gphi=0.05 edge at the
   lowest kappa. The box does not move (frozen by pre-registration); any revision is future work
   requiring its own pre-registration and fresh seeds.
4. RESIDUAL V4-LIMITED TERRITORY, explicit: (kappa=0.03, gphi=0.065) at 1/3 seeds; far-M hard
   realizations (seed 28, both M). Whether V6 would survive far-M seed 28 is UNTESTED -- no claim.

CANONICAL ENVELOPE STATEMENT (per protocol): V7 introduces no new instability beyond its frozen branches.
However, its guaranteed stable domain is limited by residual V4 failures outside the predeclared
memory-routing box.

FINAL CLAIM (earned, precise per red-team fix #1/#5): a universal learned closure was not optimal across
the tested regimes. The key structure is a STABILITY-ACCURACY FRONTIER, not a universal instantaneous
failure: accuracy-competitive instantaneous closures (the expressive learned MLP closures V3/V4/V5, and
the tested kappa4=0 truncation on one seed) DESTABILIZE in the low-decoherence (low-kappa, low-gphi)
corner, while weak instantaneous closures (linear, plain 2nd-order) stay stable there but ~2x less
accurate (Result 20, finding 1). Adding memory traces -- with rates FIXED A PRIORI to the EOM's decay
rates (a no-tuning design choice; the ablation isolates memory PRESENCE, not the necessity of that
specific timing, which is untested) -- restores both stability and accuracy in that corner. A predeclared
physics-routed closure selects the trace-bearing model only in the mapped (kappa, gphi) region (at
g=0.7, sigma=0.15, M=5, gdown=0.02, t<=40), preserving V4's broader accuracy while recovering the
trace-bearing model's stability domain -- without retraining or outcome-based selection. ARCHITECTURE
WORK ENDS HERE (pre-registered stop rule).

## Result 20 — the baseline ladder (dev + T3): where each rung fails, and what that buys the thesis

All baselines trained on VERIFIED-identical rows to V5/V6 (row-identity gate vs saved contributions);
same references, same regions, per-region reporting only. Dev seeds 29/31; final 32/33/34 in progress
(the final's M=5 regions completed pre-disk-crash and reproduce the dev picture; far-M completion and the
T3 final pass rerun next).

| rung | in-box stab / med | sanity med | far-M stab / med | failure mode |
|---|---|---|---|---|
| linear L21 / L27 | 100% / 0.187, 0.181 | 0.182, 0.131 | 100% / 0.178, 0.174 | fails SOFT: stable everywhere, inaccurate everywhere |
| sparse quad Q21 / Q27 | 100% / 0.105, **0.079** | 0.080, 0.074 | 100% / 0.216, 0.313 | accurate-ish pocket, weak normal + far-M |
| T3 (kappa4=0, gated 1e-14) | 0.37-0.38 deep corner | 0.43-0.49 normal | not runnable (M<=4 cap; 352 tracked at M=4) | expensive AND inaccurate (see below) |
| V4 | 50% / 0.091 | **0.030** | 0% (these seeds) | pocket + far-M realization-risk |
| V6 | **100% / 0.053** | 0.039 | **100% / 0.138** | +~0.03 sanity cost (known) |

Key findings:
1. **Failure modes are asymmetric.** Simple closures fail SOFT (linears: 100% stable everywhere, 0.13-0.19
   NRMSE everywhere -- weak corrections cannot destabilize but cannot capture the physics); complex
   closures fail HARD (instability at support boundaries). V7's claim is threading exactly that needle.
2. **The tested kappa4=0 third-order truncation is the ladder's decisive rung** (precise wording,
   artifact: results/t3_dev_table.md): explicitly tracking third-order moments was substantially more
   expensive (3.7x state variables at M=4; minutes-scale dense builds) and still much less accurate than
   V7 across every tested cell -- 0.37-0.38 in the deep corner (vs V7's 0.08-0.16) and 0.43-0.49 in
   normal cells where it beats plain second-order barely or not at all (0.49 vs 0.42 at M=4). Every T3
   engine passed its derivative gate at ~1e-14: this is the method failing, not the implementation. NO
   claim that all higher-order cumulant methods fail -- one specific conventional closure was tested, and
   it lost on both cost and accuracy.
3. **The sparse quadratic is the interpretability bridge**: Q27 (289 terms, 2-minute fit) reaches within
   ~1.5x of V6 in the pocket at 100% stability -- the long-coherence correction has largely compact,
   near-quadratic structure in trace-augmented local variables -- while ordinary-regime precision is
   where the MLP's depth pays (0.074 vs 0.030).
4. **Cumulative far-M stability ledger** (updated): V6 14/14 lifetime; V4 ~13/20 (0/4 on dev seeds
   29/31). V7 routes far-M to V4, so V7 inherits that realization-risk: an emergent limitation of the
   frozen router, reported, not patched (architecture closed; any routing revision is a new
   pre-registered cycle).
5. Ladder order confirmed so far: MF (frozen for these ICs) -> 2nd order -> T3(kappa4=0) -> linear ->
   sparse quadratic -> V7 -> exact, with the caveat that T3 vs linear ordering depends on region (T3
   better than linear in the deep corner, comparable-to-worse elsewhere at far higher cost).

FINAL (untouched seeds 32/33/34) -- the dev picture replicates, with two additions:
| rung | in-box stab / med | seam+resid med | sanity med | far-M stab / med |
|---|---|---|---|---|
| L21 / L27 | 100% / 0.170, 0.164 | 0.171, 0.145 | 0.171, 0.130 | 100% / 0.175, 0.171 |
| Q21 / **Q27** | 100% / 0.097, **0.078** | 0.085, **0.058** | 0.087, 0.079 | 100% / 0.168, 0.216 |
| V4 | 66.7% / 0.040 | 0.034 | 0.029 | 100% / 0.107 |
| V6 | **100% / 0.042** | 0.037 | 0.041 | 100% / 0.115 |
- **T3 destabilized on a final seed** (M=4 deep corner, s34, gate 1.0e-14): the tested kappa4=0
  truncation is not even reliably stable in the long-coherence corner. Its full T3-final table:
  deep corner 0.37-0.50 + one X (vs V7 0.06-0.16); normal 0.39-0.49 vs plain 0.42-0.45.
- **Q27's compact-quadratic pocket result replicates** (0.078 vs dev 0.079; ~1.9x V6) -- dev+final.
- far-M on these seeds: ALL models stable incl. V4 6/6 (the realization coin-flips landed favorably this
  round). Cumulative far-M ledger: V6 20/20 lifetime; V4 ~19/26.
- Rollout wall times (s, mean): V4 9.5 (inflated by its in-box divergences), V6 2.8, linears ~2.2,
  quadratics ~4.7.

## Result 21 — interpretability verdict: structure identified, no compact law; V7 stands as a validated
## computational closure (pre-registered outcome 4, in its pre-written words)

WHAT THE STRUCTURE ANALYSIS FOUND (contribution ranking along closed-loop dynamics x config-resampled
stability selection; both tables highly reproducible -- dozens of terms at selection frequency 1.00):
1. Family decomposition by region (dominant type): deprived of traces, Q21 substitutes REGIME-scaled
   statics in the hard regions (d/kappa, g/kappa interactions dominate its box/residual budgets) and its
   instantaneous terms lead only where physics is easy -- and that substitution costs accuracy. Given
   traces, Q27 leans on them everywhere (h_nc leads every region's top-10), with the collinearity caveat
   (h_nc ~ nc/M in slow dynamics) explicitly noted; stability selection kept both at freq 1.00, so the
   preference is robust rather than a Lasso trade, but this remains CONSISTENT WITH a history-dependent
   correction -- not proof of a formal memory kernel.
2. THE FORMULA LADDERS FAILED IN ROLLOUT (dev 35/36, locked rule: 100% stability + >= 70% advantage
   share): branch N best rung share 0.02; branch B rungs 5/10 NEGATIVE (-0.13/-0.18 -- partial formulas
   actively harm) and rung 20 only 0.10. All rungs stable; none close. Per the locked rule, NO formula
   was frozen and seeds 37/38/39 were never consumed.
3. THE SHARPENED FINDING: the full 289-term sparse quadratic bridge (Q27) preserves ~90% of V7's box
   advantage in rollout (0.078 on final ladder seeds), while <= 20-term distillations of ITS OWN most
   stable, most contributing terms collapse to <= 10%. The third-order correction is REPRESENTABLE in a
   quadratic basis but IRREDUCIBLY DISTRIBUTED across ~10^2 terms in these variables. Bounded claim: NO
   COMPACT 5-20 TERM FORMULA IN THE TESTED SYMMETRY-RESPECTING QUADRATIC LIBRARY preserved V7's rollout
   advantage -- not "no simple law exists" broadly; other libraries/variables remain untested. Partial
   truncations of the quadratic were worse than nothing.
FINAL WORDING (per the pre-registered tree, outcome 4): V7 is a VALIDATED COMPUTATIONAL CLOSURE, not a
discovered compact law. The mechanism statement remains bounded: instantaneous nonlinear structure
suffices in ordinary regimes at bridge-model fidelity; the long-coherence regime's correction is
consistent with history dependence (trace-bearing models repair it; instantaneous ones do not), and it
resists compact symbolic compression. INTERPRETABILITY IS CLOSED.

## SCOPE, CONDITIONING, AND LIMITATIONS (red-team consolidated; every claim above is bounded by these)

Every result is INTERNAL to one defined model, one 2D slice, one observable set, one initial-condition
family, one time horizon. The exact references solve the IDENTICAL model, so all comparisons are exact at
every tested point -- but the following bound what the results mean physically. A judge should be handed
this list, not have to find it.

1. MODEL (red-team #2, #7, #11). RWA Tavis-Cummings with bare-mode local Lindblad dissipators. Claims are
   about closure methodology internal to this model; exact refs solve the same model. For g/wc >~ 0.1 the
   model is NOT quantitatively representative of a physical cavity (counter-rotating terms and
   dressed-state dissipation omitted), yet g_coll is swept to 1.0 (train) / 1.1 (extrapolation): the g
   sweep tests closure robustness across the model's dynamical regimes, NOT experimental realism at those
   couplings. Single cavity mode; static Gaussian frequency disorder only (other draw_disorder families
   are implemented but never run); uniform couplings (the scalable engine's locality/permutation proof
   assumes uniform g -- coupling disorder would need an engine extension). Rates gdown = gphi baseline =
   0.02 wc are MODEL-SCALE choices compressing all dynamical regimes into a tractable window, not a real
   material's parameters; kappa/wc in 0.03-0.4 spans plasmonic-to-low-Q, but the emitter rates are far
   above real molecular radiative/vibrational values relative to wc. Mapping to measured VSC/exciton-
   polariton parameters is declared future work (frontier item), addressed qualitatively in VSC-MAPPING.md.
2. THE "BOUNDARY" IS A 2D SLICE (red-team #6, #4). Everything called a "failure boundary" is a
   (kappa, gphi) slice at FIXED g=0.7, sigma=0.15, M=5, gdown=0.02, fully-inverted IC, t<=40. It is not a
   surface in the full >=5D regime space. gdown = 0.02 is the only decoherence channel NEVER swept; the
   coherence floor gdown/2 + gphi never drops below 0.03, so "long coherence" is bounded by an un-swept
   knob. Prefer "low-(kappa, gphi)-decoherence corner at fixed gdown" over unqualified "long-coherence".
3. TIME HORIZON (red-team #3). All stability is over t <= 40 = >=1.2 cavity lifetimes at the slowest cell
   (kappa=0.03), >=4 at kappa=0.1. Observed divergence onsets (t~26-39) hug the window edge, so
   longer-horizon behavior is UNCHARACTERIZED; horizon extension is future work.
4. OBSERVABLES (red-team #8). All accuracy numbers are one-time collective expectation values: photon
   number n(t) (~ cavity output flux / kappa) and mean inversion. The tracked second-order moments are
   never error-scored individually, and NO two-time observable (emission/transmission spectrum, g2 via
   quantum regression) is computed -- yet spectra are the polaritonics observable. "(retracted: see SCOPE #4)"
   is retracted; two-time observables are an unevaluated extension.
5. INITIAL CONDITIONS (red-team #10). All training and headline claims use diagonal product states (fully
   inverted f=1; f=0.75 diagonal stress). No coherently seeded, driven, or entangled IC was run. Claims
   hold for the incoherent diagonal-product-state family; driven/coherent dynamics (where mean-field is
   live) are OUTSIDE the validated envelope -- a coverage boundary of equal standing to the zero-detuning
   one. The MF "frozen rung" is specific to these ICs (stated already in Results 10/16).

Superseded-note (red-team #9): Result 2's early per-triple reading ("the correction vanishes where the
closure would be useful (large M)") underestimated the CLOSED-LOOP consequence: the dropped-term COUNT
grows ~M^3 and errors integrate along the trajectory, so plain-cumulant TRAJECTORY error grows with M
(Results 8-9: 1.05 -> 1.30 -> 1.97 -> 2.50) even as each individual c3 shrinks. Results 8-9 supersede the
Result 2 "narrow intermediate-M window" guess.

## Result 22 — memory-criterion Step 1 (g_c sweep): the single ratio is FALSIFIED out-of-distribution;
## the direction survives. Per the pre-registered kill rule, the single-ratio claim is STOPPED.

The decider: does the memory-needed boundary move with collective coupling g_c as D_a = 4g_c/(kappa+gdown
+2gphi) predicts, with D_crit FROZEN on the g_c=0.7 slice? Predicted s_crit(g_c) = 4g_c/D_crit - gdown =>
{0.4: 0.043, 0.7: 0.085 (calibration), 1.0: 0.130}. Exact mesolve (M=5), fresh seeds 40/41/42, raw panel
in results/step1_raw_panel.csv (270 rows). Read against the pre-registered decision tree:

- g_c=0.7 (calibration): clean boundary s_crit=0.085, balanced acc 0.92, FNR 0.07. (This is the fit.)
- g_c=0.4: PREDICTION FALSIFIED. In the closure's valid support (kappa>=0.03) memoryless V4 is adequate
  EVERYWHERE -- no memory-needed region. The clean cell BELOW the predicted boundary (s=0.04,
  kappa=0.03/gphi=0.005) is adequate (V4 err ~0.06), directly contradicting "s<0.043 needs memory". The
  apparent observed s_crit=0.15 is an ARTIFACT: the kappa=0.02 row (below training min 0.03) shows V4 err
  0.20-0.48 at HIGH s (0.09-0.15) on seeds 41/42 -- closure extrapolation off the loss axis, at LOW D,
  OPPOSITE to the criterion. Excluded as generalization noise.
- g_c=1.0: direction-consistent but confounded and quantitatively off. Boundary moved UP to ~0.16
  (stronger coupling -> memory over a wider region: correct direction), but predicted 0.13 (superlinear
  miss), and confounded by 32 class-B cells where V6 ITSELF is unstable at low-loss+strong-coupling
  (kappa<=0.02: V4 err 1.3-1.7, V6 UNSTABLE) -- a combined-generalization wall.

ROBUST FINDING (retained, corrected wording -- do NOT overstate): across the CLEAN reference-supported
slices, the observed need for memory was consistent with increasing collective coupling; HOWEVER the
strong-coupling slice (g_c=1.0) was partly limited by V6 out-of-distribution instability and is NOT used to
validate a universal scaling law. Clean boundaries: {none-in-support (0.4), 0.085 (0.7)}; g_c=1.0 is
partly-confounded and is NOT a clean physical-boundary measurement. So this is NOT a validated "monotonic
memory demand across a 2.5x coupling range" claim -- only a directionally consistent trend on the clean
slices. FALSIFIED is the specific once-calibrated parameter-only ratio D_a as a boundary-location predictor
(fails at g_c=0.4; does not follow the frozen scaling at 1.0).

VERDICT (pre-registered kill rule: "if the g_c boundary does not collapse under the frozen ratio, stop"):
single-ratio D_a claim STOPPED. NOT a validated parameter-only criterion. V7's (kappa,gphi) box stays a
validated engineering router; NOT promoted to a physical law.

FUTURE PRE-REGISTERED HYPOTHESIS (not fit here -- would be post-hoc): the superlinear g_c-dependence
(s_crit ~1.9x for a 1.43x g_c rise) is more consistent with a Purcell-like burst rate ~g_c^2/kappa than the
linear Rabi 2g_c in D_a. Testing D_c = (4g_c^2/kappa)/(kappa/2+gdown/2+gphi) requires FRESH g_c/seeds with
the candidate frozen before runs. Also newly exposed: V6 has its own OOD failure wall at
strong-coupling + very-low-loss -- a real limitation to document.

## Result 23 — Theory triage rejects a universal single-ratio memory criterion

We tested whether an analytically motivated, once-calibrated parameter-only score based on collective coupling
and transverse decoherence could predict, before rollout, when the frozen memoryless closure V4 would become
inadequate and the memory-aware closure V6 would be required.

The candidate score was calibrated only on the baseline coupling slice and tested without refitting at weaker
and stronger collective coupling. It failed the preregistered out-of-distribution criterion: at weak coupling,
V4 remained adequate inside the valid support where the score predicted a memory-needed region; at strong
coupling, the apparent boundary shifted in the expected direction but did not follow the frozen scaling and was
partly confounded by V6 self-instability at very low loss.

Theory triage showed why a simple replacement ratio is not justified. Linearization around the initially
inverted bright mode gives a collective gain rate that is approximately linear in collective coupling in the
tested good-cavity regime, while closure adequacy depends on higher-order cumulant growth, disorder
realization, observable tolerance, and model support. Therefore, no universal parameter-only criterion is
claimed.

V7 remains a validated physics-routed learned closure within its declared operating domain. Its routing box is
an empirically calibrated engineering boundary, not a universal law of open quantum dynamics.

DERIVATION FOR THE RECORD (pending independent theory check -- see below). Rotating frame at omega_c, resonant
homogeneous bright mode, inversion frozen at z=+1 (quasi-static, valid t << 1/gdown): the (alpha, P) bright-mode
2x2 has eigenvalues lambda_pm = -0.5*(kappa/2 + gperp) +/- 0.5*sqrt((kappa/2 - gperp)^2 + 4*g_c^2), with
gperp = gdown/2 + gphi. Around the INVERTED state the discriminant is a sum of squares => eigenvalues are REAL
(exponential collective GAIN, not Rabi oscillation). Gain threshold g_c^2 = kappa*gperp/2 is crossed by ~50x in
every tested cell (a superradiant burst occurs everywhere -> the memory boundary is NOT the superradiance
threshold). In the tested good-cavity regime g_c >> kappa,gperp: lambda_+ ~ g_c - (kappa+gdown+2gphi)/4, i.e.
LINEAR in g_c. Consequences: (i) D_a used the correct decoherence combination (kappa+gdown+2gphi = kappa+2gperp)
but as a RATIO g_c/s' where the true net rate is a DIFFERENCE g_c - s' (s' = (kappa+gdown+2gphi)/4), which is
one reason a g_c=0.7-calibrated threshold does not transfer; (ii) the honest derived rate is linear in g_c, so
the observed superlinear g_c=1.0 boundary CANNOT be the mean-field burst rate and is most consistent with the
class-B/OOD confound; (iii) the bad-cavity Purcell rate ~g_c^2/kappa does NOT apply here (g_c >> kappa), so
adopting it to fit the superlinear trend would be post-hoc, not theory.

PENDING (external theory sanity check, no new model work): have an outside quantum-optics person
independently verify the sign convention, the gperp = gdown/2 + gphi definition, and the eigenvalue derivation
before this linearization is treated as final physics.

PLACEMENT (permanent): do NOT lead the poster/abstract with the failed criterion. Lead with the validated V7
method. Result 23 lives in a rigor/limitations panel (and in interviews) as evidence that the stronger
universal-criterion hypothesis was tested and honestly killed. The criterion failing does not make V7 fail; it
only means no universal physical rule for choosing V4 vs V6 is claimed. CRITERION LINE: CLOSED.

## Result 24 — Physics-prediction extension (superradiant burst vs M): scaling headline FALSIFIED; a real,
## exact-validated burst-peak-overshoot of standard cumulants survives (constant, not a scaling law)

Pre-registered (PROTOCOL "PHYSICS-PREDICTION EXTENSION") BEFORE any run: does the standard 2nd-order cumulant
closure's error on a MEASURABLE collective-emission observable -- superradiant burst peak flux
F_peak = kappa*max_t<a^dag a> and delay t_peak = argmax_t<a^dag a> -- grow systematically with molecule number
M in the long-coherence pocket, and does frozen V7 recover it? Fresh seeds {50,51,52}; exact mesolve M=2..5,
Nc_ref=16, Fock guard; closure engines at the frozen V7 eval config (NC=12, Fsafe=6). Kill rules K1/K2/K3.

OUTCOME -- K1 TRIGGERED (scaling headline dead). Plain-cumulant relative burst-peak error is FLAT at ~0.28-0.30
across M=2..5 (pocket AND control), NOT growing. Per the pre-registered kill rule ("if K1 fails at M<=5, stop,
report the null, do NOT extend to mcsolve"), the "systematic-failure-scaling-with-M" claim is FALSIFIED and the
mcsolve extension was not run. No spin.

WHY FLAT (structural, not effort): collective coupling g_c is held M-independent (g_i = g_c/sqrt(M)), so the
collective burst is ~M-independent by construction and the relative closure error cannot scale with M. A true
superradiant N-scaling lives in the fixed single-emitter-g convention (collective coupling ~ g*sqrt(N)), which
pushes coupling outside frozen V7's training envelope at large M -> unreachable without retraining V7 (out of
scope under the V7-frozen constraint). The M-scaling headline is therefore structurally out of reach here.

WHAT SURVIVED (real, signed, exact-validated): the standard 2nd-order cumulant closure OVERSHOOTS the
superradiant burst peak flux by a robust +28% to +30% (signed positive; pocket seed-50 +0.278/+0.284/+0.290/
+0.291 at M=2/3/4/5; control +0.290/+0.289/+0.296/+0.303), across M and across BOTH regimes. Frozen V7
reproduces the exact peak to within +/-0.7% on the SAME cells and the SAME engine/Fock cutoff -- so the ~29% gap
is the CLOSURE ORDER, not numerics (a clean internal control: same NC=12/Fsafe=6 for plain and V7). V7's residual
STRADDLES zero (+0.003, -0.007, +0.003, ...), i.e. an accurate model with unbiased residual, not a bug copying
the reference. The error is concentrated in burst HEIGHT: plain cumulant reproduces the burst DELAY t_peak to
within one time-grid step while overshooting the peak by ~29%.

TRUNCATION-LADDER BRACKETING (kappa4=0 T3 rung, seed 50, full run): the conventional THIRD-order truncation
does NOT fix the burst peak either -- it UNDERSHOOTS, the OPPOSITE sign to 2nd-order's +29% overshoot: pocket
signed -0.277 (M=2,3), control -0.251/-0.255/-0.259 (M=2,3,4). So going one truncation order up does not
converge the burst peak; it flips the error sign while keeping ~the same magnitude. The learned closure (V7,
+/-0.7%) sits between the two conventional truncations and matches exact -- the strongest form of the finding:
the burst-peak error is a STRUCTURED correction that standard low-order truncations bracket but miss, not a
"just track higher order" artifact. Additionally, T3 is DIVERGENT in the low-loss pocket at M=4 (integration
blew up under the guard) -- the known kappa4=0 stiffness pathology -- so it is not even numerically usable
there, a further reason higher-order truncation is not the fix.

SCOPE/HONESTY: this is a CONSTANT offset, NOT a scaling law, and NOT pocket-specific (plain overshoots in the
normal-loss control too) -- so it is a general property of 2nd-order cumulant on the inverted IC, not a
memory-routing effect. It is a real, measurable, researcher-relevant result (standard cumulant overestimates
superradiant burst intensity by ~30%; V7 supplies a validated correction), but a solid poster PANEL, not a
field-changing scaling discovery. WORDING DISCIPLINE (two tiers, do not conflate): (a) CROSS-PLATFORM
REPRODUCTION = an independent person runs the QuantumCumulants.jl artifact and reproduces the ~29% overshoot;
(b) INDEPENDENT REPLICATION = (a) PLUS an independent EOM/sign-convention check, an M=2 hand-derivation, and
recorded Julia/package versions + commit + numerical difference. Only tier (b) earns the phrase "independently
reproduced." Until an outside collaborator completes (b), say only "cross-platform reproduction pending." Artifact built:
replicate_cumulant.jl, results/replication_params.json, REPLICATION.md (exact reproduction targets + the
replicator request embedded).

## Result 25 — Mechanism of the burst-peak error: a SMALL-FAMILY (two-channel) connected-third-cumulant
## correction, reconstructed by V7 (final frozen internal computation)

Pre-registered (PROTOCOL "MECHANISM ANALYSIS") BEFORE the run; frozen V7, no retrain/tune/reroute; ablations
zero a kappa3 family on the CLOSURE OUTPUT only. Fresh seeds 37/38/39; POCKET and CONTROL cells at M=4;
kappa3_exact = <k> - v3_factorized(m2_exact); GATE kappa3_exact(t=0) < 1e-6 passed every cell; burst window =
flux >= 50% of exact peak. Families = conjugation-canonical factor-type signatures.

QUESTION: is the exact superradiant burst-peak correction one dominant connected-kappa3 family, a small set, or
broadly distributed? Adjudicated by the NON-CIRCULAR D_f = ||T3 . kappa3_family|| (total dynamical influence)
and closed-loop ablation (fraction of V7 burst-peak advantage lost). [The pre-registered DIRECT driver E_f
(contribution to the collective emission-coherence derivative) is structurally single-family -- only
photon-number-inversion enters d<a sigma^+>/dt at order 3 -- so E_f is a channel identifier, NOT an unbiased
scan; disclosed. D_f and ablation are the adjudicators.]

RESULT (robust across 3 seeds AND both regimes): the correction is carried by exactly TWO connected-kappa3
families, together ~100% of D_f (all other families ~0.00):
  - DOMINANT: photon-number-inversion  <a^dag a sigma^z>  (D-share 0.65-0.67; ablation removes 0.81-0.84 of
    V7's burst-peak advantage; ablating it DESTABILIZES V7 in the pocket -> critical). V7 reconstructs its
    connected kappa3 to sign and magnitude (V7mag 0.99-1.02, sign +).
  - SECONDARY: photon-polarization-inversion  <a sigma^+ sigma^z>  (D-share 0.33-0.35; ablation removes
    0.25-0.28; also critical in the pocket). V7 reconstructs it (V7mag 0.91-1.08, sign +).
Notably the DOMINANT channel is photon-number-inversion, NOT the photon-polarization-inversion triple (that is
the SECONDARY) -- the pre-registered "do not assume the triple" caution was correct.

PRE-REGISTERED CONCLUSION: 2 = SMALL-FAMILY DISTRIBUTED mechanism. The single-family bar (D-share >=0.5 AND
>2x the second) is a threshold knife-edge here: pocket 0.67/0.33 = 2.03x (barely passes), control 0.65/0.35 =
1.86x (fails). The secondary family carries a reproducible ~1/3 of D_f and ~1/4 of the ablation effect in BOTH
regimes across all seeds, so the robust cross-regime call is TWO families, not one.

WHY THIS IS THE MECHANISM OF THE +29%/-28% BRACKETING (Result 24): the dominant connected kappa3 is large and
negative through the burst (Re kappa3 ~ -0.5 to -1.0). Plain 2nd-order sets it to 0 -> misses a large negative
cumulant -> burst-peak OVERSHOOT (+29%). kappa4=0 T3 evolves it but OVER-corrects/mis-estimates it (control: T3
dives to ~-2.2 vs exact ~-0.8; pocket: T3 divergent) -> UNDERSHOOT (-28%) / instability. V7 reconstructs it
(exact and V7 overlie in the burst window) -> +/-1%. Figure: results/mechanism_fig.png (D_f shares; dominant-
family kappa3(t) exact/V7/T3/plain; ablation frac-lost).

SCOPE/HONESTY: this is a mechanistic attribution of the burst-peak correction to two named cumulant channels in
THIS open disordered Tavis-Cummings system; it is exact-referenced at M=4 and consistent with the M-flat
finding of Result 24. It is NOT a universal claim about all open systems. The E_f structural caveat is disclosed
above. This is the FINAL internal science computation; all modeling is now FROZEN. Next: external replication
(cross-platform, then independent), manuscript, ISEF poster, whiteboard defense.

## Result 26 — Disorder-sweep kill test (candidate extension): NULL. Standard-method failure is disorder-robust
## at fixed G; disorder only monotonically redistributes the connected correlations. Third killed extension.

Pre-registered candidate hypothesis (honest framing; NOT a discovery): does spectral disorder sigma produce a
non-trivial (ideally non-monotonic) change in the two Result-25 connected-third-order channels and make the
2nd-order closure meaningfully more wrong, tied to a burst observable? Exact + plain 2nd-order ONLY (no V7),
fixed g_c (=> G^2 fixed), kappa=0.03, gphi=0.02, full inversion, M=3,4, held-out seeds 60-64, sigma in
{0..0.45}, mean +/- SEM over 5 draws.

RESULT: KILL. Every RESOLVED quantity (signal/noise>3) is MONOTONE; the sole non-monotonic flag (kappa3 total,
M=4) is noise (s/n 1.7). Across sigma 0->0.45: F_peak monotone DOWN ~16% (disorder suppresses the burst --
expected); t_peak flat (s/n 1.1, unresolved); kappa3 photon-number-inversion DOWN ~15-20%; kappa3 photon-
polarization-inversion UP +66-83% (s/n up to 12.8 -- the most robust effect, but monotone); kappa3 TOTAL ~flat
(conserved to ~2-7%); 2nd-order F_peak error UP only ~3 points (28->31%, monotone). So disorder does NOT expose
a new failure of standard cumulants -- the standard-method burst-peak failure is essentially DISORDER-ROBUST at
fixed G.

The one genuine (minor) finding: disorder CONSERVES the total connected third-order correlation but
REDISTRIBUTES it between channels (number-inversion -> polarization-inversion; cleaner at M=4, split shifts
~86/14 -> ~72/28 by sigma=0.30). Mechanistically sensible (disorder activates the polarization channel that
couples bright<->dark), but monotone and observable-consequence-free -- "a plot, not a contribution." NOT a
headline. Scripts disorder_sweep.py / disorder_report.py; figure results/disorder_sweep_fig.png.

STANDING NOTE (three killed extensions -- 22/23 universal criterion, 24 M-scaling, 26 disorder): the "make it
more novel" extension well is dry. The real OPEN vulnerability for the CORE result is NOT disorder but whether a
standard EVEN-order (4th/6th) cumulant closure would reproduce the burst peak (we tested only the known-weak
kappa4=0 odd/third rung); until that is closed, "learned closure necessary" is unsupported. That baseline, plus
external replication + manuscript, is the work -- not further novelty hunts.

## Result 27 — Fourth-order cumulant fairness test: no TESTED zero-next-cumulant truncation through 4th order
## gave a stable, accurate burst closure in the evaluated cells (kappa5=0 DIVERGES); V7's practical value
## established, not field-defining. [WORDING: never say "no working standard truncation exists" -- only that the
## TESTED zero-next-cumulant truncations through 4th order failed in the EVALUATED disordered open-cavity cells.]

The fairness question a referee will ask (PROTOCOL "FOURTH-ORDER CUMULANT BASELINE"): would a properly-built
fourth-order cumulant closure (track to order 4, set connected kappa5=0) just solve the burst peak, making a
learned closure unnecessary? Built T4Engine one order up from the validated kappa4=0 T3 engine, via the general
set-partition moment-cumulant closure (VERIFIED to reproduce T3's hardcoded kappa4 terms at n=4). Unforgeable
gate PASSES at ~1e-14 (assembled d<k>/dt == exact <L^dagger[k]> on all-exact moments) -> the engine is correct.

RESULT (outcome O3, strongest for V7): the kappa5=0 fourth-order truncation GENUINELY DIVERGES -- finite-time
blow-up at t~9.65, M=3, BOTH regimes. This is NOT a solver artifact: FOUR integrators (LSODA at rtol 1e-5 AND
1e-8, plus stiff implicit Radau and BDF) all diverge at the IDENTICAL time and magnitude (8.8e5). It also
OVERSHOOTS the peak (n~2.8 vs exact 2.31 at t=3) before diverging, and is EXPENSIVE (~100s to build at M=3;
348 tracked moments; construction cost large and highly variable even at M=3). So among the TESTED
zero-next-cumulant truncations through 4th order, NONE gave a stable, accurate burst closure in the evaluated
disordered open-cavity cells:
  - 2nd-order:   OVERSHOOTS peak +29%  (stable)
  - kappa4=0 3rd: UNDERSHOOTS peak -28% (stable in control; runaway growth triggering the guard in the low-loss pocket)
  - kappa5=0 4th: reproducible RUNAWAY GROWTH triggering the divergence guard in all evaluated M=3 cells, and
                  overshoots the peak before the runaway
  - V7:          +/-0.7%, stable; avoids high-order engine construction
HYPOTHESIS (not tested; discussion only): why this differs from the spin-only free-space Dicke benchmark, where
4th order becomes exact at N=4 -- our model has an unbounded cavity Fock mode, and high-order cumulant truncations
of a bosonic mode MAY be prone to this runaway behavior, since a finite spin Hilbert space closes at a bounded
order while a Fock mode does not. This is a discussion point, not an established mechanism.

WHAT THIS SETTLES (and what it does NOT): it settles the fairness/necessity question FOR THE TESTED TRUNCATIONS
-- none through 4th order gave a stable, accurate, practical burst closure in the evaluated cells, so V7 is a
MEANINGFUL PRACTICAL CLOSURE for this regime, not a fancy substitute for a working standard method. It does NOT
claim no such method could exist, and does NOT make the project field-defining. Scope: kappa5=0 tested at M=3
(M=4 not built -- construction was substantial and highly variable already at M=3; M=3 already settles the outcome).
Scripts: t4_truncation.py, t4_eval.py,
t4_diag.py; diagnostic results/t4_diag.log. ROBUST ACROSS HELD-OUT DISORDER REALIZATIONS (deterministic draws
50/51/52, NOT stochastic-solver validation): T4 kappa5=0 unstable in ALL 6 M=3 cells (both regimes); plain
+0.284/+0.289, T3 -0.277/-0.256, V7 -0.001..-0.007 -- consistent across draws. T4 construction/rollout costs were
rapidly increasing AND highly variable even at M=3 (build 72-1068s; failed rollouts up to ~1056s) -- impractical.
(The O(M^5) growth is the expectation; a build-scaling table across multiple M was NOT measured.) Grid:
results/t4_eval.log / results/t4_eval.npz.

## Bottom line of the whole spike
Every claim is backed by validated code in this folder. The chain is COMPLETE: a QuTiP-validated exact engine;
a second-order cumulant engine validated to ~1e-14; a demonstrated failure of plain cumulant that the 3rd
cumulant provably fixes; a learnable, permutation-equivariant closure map (R2 up to 0.99); a STABLE closed-loop
closure; a SCALABLE engine (Result 7) proven identical to machine precision (max diff <= 5e-14) to the dense engine at M=2..5 that runs the closure
at arbitrary M; and the closed loop of the method itself (Result 9): retraining on the M the engine unlocked
recovers accuracy at M beyond it (2.8x over plain cumulant at M=10, relative error improving with M). Rollouts
at M=12-20 are stable and physically shaped but exact-UNVALIDATED: numerical reach, not demonstrated accuracy.

The honest score: a strong 7, arguably 7.5 now. The full arc -- exact physics, validated engines, a
demonstrated failure mode of a real approximation method, an ML closure that fixes it, a scaling method that
breaks the exact-simulation wall, and a demonstrated bootstrap one rung up the M-ladder -- is a defensible,
method-contribution-level piece of work with every number gated. It is still one regime, still a model system,
and the absolute large-M accuracy still has headroom: not an 8-everywhere miracle, and it does not need to be
dressed up as one.

## The open frontier (what turns this into the full project / paper)
1. **Climb the next rung.** Use mcsolve (or MPS) ground truth at M=8-12 as training data (the targets carry MC
   noise, so this needs either bigger ntraj or noise-robust regression) and validate at M=16-20 the same way.
2. Positivity/physicality constraints on the closure (the z_end < -1 overshoot at M>=12).
3. Robustness across coupling/loss regimes and disorder families (only one regime tested here).
4. Map to real molecular VSC parameters and connect to a measurable observable.

## Files (all re-runnable)
exact_tc.py, validate_model.py (solver + gates) | experiment2/3/4.py (physics diagnostics) |
gen_closure_data.py, learn_closure.py (learnability) | mean_field.py, cumulant.py (validated dense engine) |
closed_loop.py (plain vs oracle vs exact) | learned_loop.py (global closure) |
local_closure.py (stabilized transferable closure) | transfer_scan.py (M-transfer, Result 6) |
step1_locality.py (locality proof) | scalable_engine.py (arbitrary-M engine, Result 7) |
validate_scalable.py / validate_stream.py / validate_compiled.py (engine validation vs dense) |
mc_ref.py (quantum-trajectory reference, cached) | check_closure_vec.py (vectorized-closure check) |
gen_data_v2.py (dense-engine-free exact data at M=4,5) | check_eops_path.py (e_ops == dense-path proof) |
retrain_eval.py (Result 9 head-to-head) | plot_retrain.py (figure + pickled closures) |
local_run.py (large-M flagship run, Result 8) | results/*.png, results/*.log, results/*.npz
