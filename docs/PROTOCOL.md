# Pre-registered protocol for the regime-generalization campaign (v3)

Written BEFORE any v3 training. Success criteria fixed here; results are reported against them whether
they pass or fail. Baseline for all comparisons: VERSION0.md (git tag v0).

## AMENDMENT (post-campaign, at external review; statistics recomputed from saved cells, no retraining)
Claim classes are THREE-way, by what the cell actually holds out:
  D  in-distribution: regime tuple ON the training grid, disorder seed unseen (new-realization
     generalization -- the weakest class, never reported as interpolation)
  I  interpolation: tuple inside the hull but never trained, seed unseen
  E  extrapolation: at least one parameter outside the hull
Every reported row carries provenance: tuple trained? seed trained? IC trained? M trained? reference type.

## Claims are evaluated SEPARATELY (never blended into one aggregate)
1. INTERPOLATION: held-out regime combinations inside the sampled hull (unseen (g,k,gphi,sigma) tuples,
   unseen disorder seeds), at M inside the training range.
2. PARAMETER EXTRAPOLATION: withheld regions outside the training hull (high dephasing gphi=0.2-0.3,
   high disorder sigma=0.4, weak coupling g=0.3, strong coupling g=1.1, low/high kappa 0.03/0.4).
3. SIZE EXTRAPOLATION: M in {8,10} (never trained), exact-referenced by seeded mcsolve; M>10 reported
   only as unvalidated numerical reach.
4. INITIAL-CONDITION TRANSFER (stress test): partial inversion f=0.75 mixed product state, standard
   regime; trained only on fully-inverted data.

## Metrics (every method, every test point, same disorder realizations = paired)
- NRMSE per observable = RMSE(obs - exact)/RMS(exact); observables: n(t), mean sz(t)
- max-abs error (continuity with V0 tables), peak error |n_peak - n_peak_exact|, time-to-peak error
- stability rate: fraction of test points with finite, bounded (|m|<1e2), full-length rollouts
- absolute values ALWAYS shown next to ratios

## Statistics
- Paired per test point (identical eng/w/g per method). Campaign-level claim: paired bootstrap (10k
  resamples) 95% CI of median NRMSE improvement over plain 2nd-order cumulant must exclude zero.
- Test seeds: 7, 8, 9 minimum for headline rows. Single-seed results are illustrative only.

## Baselines at every test point
mean-field, plain 2nd-order cumulant, learned v2 (regime-blind, frozen), learned v3.
(Conventional 3rd-order truncation: separate workstream; must not delay the campaign.)

## Pilot gate (runs BEFORE the full 432-config campaign)
Pilot: corners+center of the regime grid, seed 0 only, M={2,3,4} (18 cfgs) + M=5 corners (8 cfgs), ~10 min.
Two feature representations trained on identical pilot data (raw regime tuples stored; transforms applied
at train time, bijective at kappa>0 so information content is identical):
  RAW:   [.., site detunings Delta_i, .., gcoll, kappa, gphi]
  RATIO: [.., Delta_i/kappa, .., gcoll/kappa, kappa, gphi/kappa]   (kappa kept raw: time is lab-frame)
Rationale: lab frame with wc fixed => equivalent for interpolation; ratios can only matter for
extrapolation. Decided empirically on extrapolation cells, not assumed. Note the ratio transform must
include the detunings, or it is an inconsistent hybrid.
DECLARED COVERAGE BOUNDARY: global ensemble detuning mean(w)-wc = 0 in ALL v3 data (disorder centered on
the cavity). Any future detuned-ensemble claim requires new training data and its own extrapolation tests.
Proceed to full campaign only if, on pilot test points:
  (a) v3 (either representation) beats plain cumulant on median NRMSE at interpolation points;
  (b) stability rate 100% at interpolation points;
  (c) the better representation on extrapolation cells is selected for the campaign;
  (d) if BOTH representations are worse than plain cumulant on extrapolation cells, STOP: fix the
      representation before generating the full dataset.

## Pilot decision table (each row judged on ITS OWN cells; no cell-averaged headline anywhere)
| Test claim | Required evidence |
|---|---|
| Interpolation | learned closure beats 2nd-order cumulant on held-out parameter combinations |
| Parameter extrapolation | improvement survives held-out boundary/corner cells (paired) |
| Size extrapolation | improvement survives completely unseen M with exact references where feasible |
| Physical validity | stability rate not worse than baseline; no new major bound violations |
| Representation choice | raw vs ratio decided specifically on extrapolation cells, not average score |
Guard: a model that wins easy cells but loses the high-disorder/high-dephasing cells FAILS the campaign
gate; those cells are the project's justification.

## Presentation rule: the physics hierarchy
Results are framed as a correlation ladder, not four curves:
  mean-field -> 2nd-order cumulant -> learned 3rd-cumulant closure -> exact reference
Each rung restores missing correlations; the learned closure supplies the leading missing piece without
carrying the full higher-order system.

## Reporting rules locked before the 2x2 / staircase / support-risk verdicts
1. SUPPORT-RISK: all numbers fixed HERE, before any risk trace is computed (disclosure: divergence onset
   times from the seed-7/8 probe were already known when these rules were set; no risk-score output had
   been seen). Threshold: d/d0 >= 1, with d0 the per-type 99th percentile of training self-distance
   (fixed at fit time, no tuning). Trigger time: first integrator call with d/d0 >= 1. Material lead:
   lead = t_onset - t_trigger >= 2.0 model time units (= 10 recorded steps = 5% of TMAX). Success
   criteria: (a) lead >= 2.0 in at least 80% of unstable dev rollouts (seeds 7/8); (b) false-alarm
   control: on the three stable Q1 center-regime rollouts (in-support), time-fraction with d/d0 >= 1
   must be < 5%. An indicator that only spikes once the rollout is already broken, or that fires
   constantly in-support, fails.
2. STAIRCASE (and any stability-affected table): NEVER average stable and unstable runs into one error
   number. Stability rate and NRMSE-conditional-on-stability are shown side by side; blowups are marked,
   not dropped.
3. 2x2 winner: chosen on seeds 9/10 only, against all four criteria jointly (stability rate, conditional
   NRMSE, P(r>0.9)/P(r>0.99) saturation, max r) plus the out-of-pocket sanity set; frozen; final verdict
   on untouched seeds 11/12/13. Decision tree (exactly one outcome): both-arms wins -> freeze density+gate
   as V4; density wins -> density-only V4; gate wins -> gate-only V4; neither wins ALL criteria -> NO
   WINNER, the minimal memory experiment is next. A method that stabilizes but loses accuracy is not a
   winner. If a V4 is frozen: before anything else, prove the boundary MOVED (re-run dense pocket grid,
   M=6-10 staircase, sanity cells with frozen V4; V3-vs-V4 side by side).

4. SEED LEDGER (binding; amended after Result 15, before any fresh-seed V4 data): 7/8 diagnosis+dev.
   9/10 model selection (burned). 11/12/13 final verdict for V3-vs-V4, BOTH axes (pocket grid AND the
   M=6-10 staircase) -- never reused for tuning. 14/24 V5 development/selection. 16/17/18 V5 final
   verdict. 19/20 memory-model development. 21/22/23 memory-model final verdict.
   COLLISION AMENDMENT (dated: caught by adversarial review before any V5/V6 data existed): derived
   training rng seed train_rng_seed(0, 0.15) = 15 equals former ledger seed 15 (identical disorder
   realization at sigma=0.15). Seed 15 is RETIRED from the ledger, replaced by 24; disjointness of the
   full derived set against the full ledger is now asserted in code at generation time.

4b. V4 STATUS + V5 PRE-REGISTRATION (dated: after Result 15, before fresh-seed results). V4 is a
   CANDIDATE under held-out validation, not a confirmed winner: it improved the targeted pocket on
   selection seeds but regressed size transfer on the comparison seeds (a pocket specialist until fresh
   seeds say otherwise). The 2x2 selection was amended post-hoc (max-r degeneracy), so the fresh-seed
   results carry the credibility. Scorecard: TWO-AXIS ONLY (pocket stability/error; size stability/error),
   no combined average.
   V5 = density-COMPENSATED training. Identical data pool (train_v3 + train_pocket), identical
   architecture, optimizer, initialization policy (random_state=0), and fit configuration; the ONLY
   intended change is area-weighted subsampling in the (kappa, gphi) plane so each grid point's influence
   is proportional to the parameter area it represents (pocket keeps resolution via a floor of 2,000
   samples per (kappa,gphi) point per type, loses numerical domination; sklearn MLP has no sample
   weights, so compensation is by subsampling).
   EXPOSURE MATCHING (locked before fresh verdicts): because area-compensation shrinks totals and no
   additional main-grid samples exist, V4-vs-V5 cannot differ in mix alone at matched N. Therefore a
   third fit, V4' = V4's natural mix subsampled to V5's exact per-type totals, is trained alongside:
   V5 vs V4' differ ONLY in mix at equal sample count; V4 vs V4' isolates count. All fits share
   max_iter/early-stopping config (number of updates is data-dependent under early stopping; disclosed).
   The ACTUAL per-(kappa,gphi)-point sample contribution per type is saved with each trained model.
   Question answered: did V4 regress because pocket data is harmful, or because the pocket was
   OVERWEIGHTED, or merely because totals changed?
   TERMINOLOGY: this is a "measured coverage-transfer tradeoff" -- never "tradeoff surface" (broader than
   the experiments establish).
   Decision tree on fresh V4 results: wins pocket + loses size -> V4 is a localized-density result; run
   V5. Wins both -> staircase regression was seed-sensitive; V4 leading candidate (still reported).
   Fails fresh pocket -> density not robust; V3 stays the method; go to memory. V5 improves pocket
   without harming size -> the upgrade. V5 still trades off -> memory has earned its experiment
   (kappa 0.03-0.035, weak gphi, M=9-10, dev seeds 19/20 only).

5. SUPPORT-RISK: frozen as a NEGATIVE result (Result 13). Honest sentence: "the support-risk signal
   detected failures early but triggered too often during valid in-support dynamics, so it failed the
   safety-system criterion and was rejected." Methods/interview material, not a poster headline. No
   further tuning now.

## V6 PRE-REGISTRATION (memory-aware closure; dated before the fresh two-axis scorecard is assembled)
MOTIVATION (stated precisely): the exact closure term is a time-nonlocal functional of lower moments; an
instantaneous closure is its Markovian approximation, which degrades exactly where decoherence rates are
small (slow memory-kernel decay) -- the mapped failure corner. The seed survive/diverge split is
CONSISTENT WITH history-dependence but does not prove it (states differ between seeds); burst phase is
partially visible in instantaneous (nc, z), so failure of V6 is a genuine possibility.
DESIGN (one variant, no zoo): V6 = the V3-RATIO architecture + exponentially weighted memory traces
h_x' = lambda_x (x - h_x) for exactly three signals: nc/M, mean z, mean |coh|. Features added per trace:
(h_x, x - h_x) -> +6 features (27 total). Physical timescales, not tuned: lambda_nc = kappa;
lambda_coh = gdown/2 + gphi (the EOM's transverse decay rate); lambda_z = gdown. Traces initialized at
their t=0 values (no fake transients); in the loop, traces are integrated as additional ODE state.
CONTROL ARMS (confound-proof): V6 is trained on area-balanced data (V5 recipe). Therefore V5 (identical
balanced data, NO traces) is ALWAYS trained alongside as the paired ablation: V6 vs V5 isolates memory;
V5 vs V3/V4 isolates balancing. V5 is a control, not a bet to optimize.
SEEDS: dev/selection 19/20. Final verdict 21/22/23. (14/15 and 16/17/18 remain V5's if the branch makes
V5 a candidate in its own right.)
SUCCESS (ALL required): higher stability in the deep long-coherence pocket (kappa<=0.035, gphi<=0.05);
conditional NRMSE <= incumbent; no material sanity regression; M=9/10 stability >= incumbent; reduced
pre-divergence saturation P(r>0.9). A memory model that fixes the pocket but breaks normal regimes LOSES.
KILL: if V6 fails these on dev seeds 19/20, it is reported as a negative result; the next escalation
(rollout-trained loss, temporal architectures) is a separate scope decision, not an automatic step.

DECISION TEMPLATE (written before any V5/V6 result exists; applied mechanically to the COMPLETED dev
table only -- partial numbers are never read as evidence):
  1. V5 beats V3, V6 does not beat V5      -> balancing mattered, memory did not.
  2. V6 beats V5 in the residual pocket AND far-M, no material sanity damage
                                            -> memory is the missing ingredient (centerpiece result).
  3. V6 helps pocket but not far-M          -> limited-memory representation helps but is insufficient
                                               for size transfer.
  4. V6 fails                               -> stop architecture roulette; document the negative result;
                                               rollout-trained learning is a LATER scope decision.
V4' is DESCRIPTIVE ONLY (count-control context); it never enters the V6 decision rule. V4 appears as a
descriptive incumbent column for far-M context; the decision rule is V6-vs-V5 (+ sanity guard).

## V7 PRE-REGISTRATION (physics-routed closure; dated BEFORE any seam or final data)
V4 and V6 are PERMANENTLY FROZEN (lc_v3dens.pkl, lc_v6.pkl; no further tuning of either). V6's status is
NOT a pure negative: the 1.5x guard was not in the original protocol, so Template 4 is not a fully
pre-registered rejection; V6 stands as a validated long-coherence repair with a quantified in-support cost.
V7(kappa, gphi) = V6 if (kappa <= 0.045 AND gphi <= 0.05) else V4.
ROUTING BOX DERIVATION (dev evidence only; seeds 21-23 and any future seeds excluded): the box is the
union of every V4/dens instability observed in development data (2x2 on seeds 9/10: (0.03,0.02),
(0.035,0.02), (0.04,0.02); V6-dev on 19/20: (0.03,0.02), (0.03,0.035), (0.035,0.02)) plus ONE grid step
of margin in kappa (0.04 -> 0.045), with gphi <= 0.05 where dev evidence also shows V6 accuracy-dominant.
The literal pre-registered deep pocket (kappa<=0.035) was considered and rejected because it leaves the
mapped (0.04, 0.02) V4 failure outside; both candidates and this reasoning are recorded here BEFORE any
V7 data exists. The box never moves after this line.
RULES: route on (kappa, gphi) only -- global inputs known before simulation; never on seed, trajectory,
reference, error, or support-risk. No blending. One model per trajectory, chosen at t=0. BRANCH
EQUIVALENCE enforced by assertion: where V7 routes to V4 its trajectory must be identical to frozen V4's;
where it routes to V6, identical to frozen V6's (V7 introduces ZERO new numerics).
SEEDS (ledger extension): 24/25 = V7 seam verification ONLY (24 reallocated from the never-consumed V5
dev slot; no evaluation has ever used it). 26/27/28 = V7 final verdict, untouched. Training-seed
disjointness: derived set {5,15,30,105,115,130,205,215,230} does not intersect {24..28}.
SEAM TEST (24/25): points just inside ((0.045,0.05),(0.045,0.02),(0.03,0.05)), just outside
((0.05,0.05),(0.05,0.02),(0.03,0.065)), deep corner (0.03,0.02), sanity cells, far-M M=9/10. Purpose:
verify code + route behave as pre-registered. The box does NOT move based on 24/25 results.
FINAL VERDICT (26/27/28), reported as FOUR regions, never blended:
  1. inside memory box     -> V7 must match V6's pocket repair;
  2. outside normal region -> V7 must match V4's normal-regime accuracy;
  3. seam points           -> no new discontinuity or unexplained degradation;
  4. all tested outside-box failures -> shown EXPLICITLY as residual V4-limited territory.
ENVELOPE RULE (correction, dated pre-verdict): a V4-side instability outside the box is not a new V7
defect but IS a V7 limitation -- it bounds V7's guaranteed stable domain and is never excluded from
judging V7's operating envelope merely because the router selected V4 there. Canonical phrasing: "V7
introduces no new instability beyond its frozen branches. However, its guaranteed stable domain is
limited by residual V4 failures outside the predeclared memory-routing box."
IF V7 FAILS: stop architecture work entirely; V4 remains the main model, V6 the domain-specific finding.

## PHASE B PRE-REGISTRATION: baselines (dated before any baseline is fitted)
ARCHITECTURE IS CLOSED: V7 frozen (tag v7-final); no V8, no routing changes, no V4/V6 tuning. Purpose of
every baseline: locate V7 on a structure-complexity ladder, four regions, no pooled winner.
SEED LEDGER (deviation from proposal, documented): proposed baseline seeds 29/30 + 31/32/33 rejected
because train_rng_seed(0, 0.3) = 30 collides with the derived training set (same rng direction trained at
sigma=0.3; same leak class that retired seed 15). BASELINE DEV = 29/31; FINAL ALL-METHOD COMPARISON =
32/33/34. Disjointness against derived {5,15,30,105,115,130,205,215,230}: verified empty.
BASELINES (all trained on THE SAME balanced rows as V5/V6 -- indices reproduced deterministically and
verified against the saved v56 contributions before any fit is trusted):
  L21  linear memoryless (Ridge alpha=1e-6, 21 cols)   -- role: is nonlinearity needed vs V4?
  L27  linear trace-aware (Ridge alpha=1e-6, 27 cols)  -- role: is nonlinearity needed vs V6 in the box?
  Q21  sparse quadratic memoryless (PolynomialFeatures deg 2 + Lasso alpha=1e-4, fixed a priori, fit on a
       seeded 300k/type subsample of the same rows)     -- first interpretability bridge
  Q27  sparse quadratic trace-aware (same recipe, 27 cols)
  T3   conventional third-order truncation (kappa4=0), dense engine, machine-precision gate required
       before any trajectory is trusted; CAPPED EFFORT: M<=4 only (the O(M^4) tracked-moment growth is
       itself part of the answer); if the gate cannot be passed within the cap, the attempt is reported.
EVALUATION: same four regions as V7's verdict (in-box, normal, seam, residual territory) at M=5, plus
far-M M=8/10 for closures that run in the scalable engine. Same exact-reference protocol (mesolve M=5,
seeded mcsolve far-M, MC err stated). Report per region: stability rate, conditional NRMSE, peak error,
fit time, rollout wall time, parameter count. NO pooled single number.
LADDER FIGURE: mean-field -> 2nd-order cumulant -> linear -> sparse quadratic -> kappa4=0 truncation ->
V7 -> exact reference. (MF rung: identically frozen for these incoherent inverted ICs -- physics fact.)
Interpretation rule: a near-V7 quadratic is scientifically valuable (interpretability bridge), not a
threat; a failing quadratic proves the closure needs the nonlinear structure. Either way V7's claim is
"smallest TESTED closure structure preserving accuracy across both regimes".

## PRE-FINAL LOCK for the baseline ladder (dated before the 32/33/34 run)
1. QUADRATIC SETTINGS: Lasso alpha = 1e-4, PolynomialFeatures degree 2 no-bias, 300k/type seeded
   subsample (rng 7) -- as pre-registered. Any adjustment may use dev-seed (29/31) evidence ONLY, must be
   documented here with reason BEFORE the final run, and no adjustment may follow the final run.
2. T3 SETTINGS: dense engine, Fsafe = 6, Nc = {3: 10, 4: 11}; rollout solver RK45 rtol 1e-4 / atol 1e-6
   (identical to every other closure's loop); gate < 1e-8 required per engine build before any rollout;
   T3 evaluated at M = 3, 4 only (pre-registered effort cap).
3. REGIONS (frozen as coded in baselines_eval.POINTS): in-box = {(0.030,0.020),(0.045,0.020),
   (0.045,0.050),(0.030,0.050)}; seam = {(0.050,0.020),(0.050,0.050)}; residual = {(0.030,0.065)};
   sanity = the standing 4 cells; far-M = M 9/10 at V0 regime. RUNTIME PROTOCOL: per-rollout wall time
   via time.time() around the integrate call, same machine (Apple M4), no concurrent heavy jobs during
   the final run; fit times and parameter counts from baseline_fit_meta.
4. LADDER FIGURE LAYOUT (locked): one figure; columns = MF, plain 2nd-order, L21/L27, Q21/Q27, T3,
   V7, exact; rows = the four regions (+ far-M); each cell shows stability% and median conditional
   NRMSE; blowups marked, never averaged; MF's rung annotated "no fluctuation seed for these ICs --
   frozen" as a physics fact, not a measured zero.
5. THE QUESTION the table answers (verbatim): which rung fails where, and is V7 the first tested method
   that survives both normal and long-coherence regimes without giving up accuracy?
METHODS-LOG NOTE (kept per review): the stale-log false fire (monitor matched a previous run's traceback)
is recorded as the infrastructure analogue of the partial-data misread -- completion markers must be
unambiguous about WHICH run produced them; chained jobs now truncate logs at launch.

## Wording rules (claims discipline)
- "accurate" only where exact-referenced (M<=10 with stated MC error). Beyond: "stable rollout,
  unvalidated extrapolation".
- Disorder/symmetry defense: full permutation symmetry is broken by emitter disorder, so identical-emitter
  reductions are not exact for this model; we benchmark against full exact methods where feasible. No claim
  that no reduced method could help.

## BASELINES CLOSED (permanent)
All ladder rungs measured, gated, seeded (final 32/33/34), figure locked (results/ladder.png). No further
baseline work; interpretability is the active phase.

## INTERPRETABILITY PRE-REGISTRATION (dated before any term analysis)
SEED LEDGER EXTENSION: 35/36 = interpretability development (formula selection); 37/38/39 = formula final
verdict. Disjointness vs derived training seeds {5,15,30,105,115,130,205,215,230}: verified empty; the
analysis scripts assert it. Final baseline seeds 32/33/34 are NEVER used to pick formulas.
RULES (locked):
1. Terms are ranked by ACTUAL CONTRIBUTION along trajectories, c_j(x) = w_j * phi_j(x_std), reported per
   region (normal / memory box / seam / residual) as median |c_j|, 95th pct |c_j|, sign consistency --
   never by raw coefficient magnitude (scale + collinearity trading make raw weights misleading).
2. Q21 and Q27 analyzed together, by term family: instantaneous collective (nc/M, z, pairs), nonlinear
   products, regime interactions (d/k, g/k, gphi/k), trace terms (h_x, x - h_x). The V6-mechanism check:
   do trace terms actually carry the pocket repair?
3. Grouped stability selection: repeated sparse fits on TRAINING data resampled by trajectory family
   (config ids, never rows); report selection frequency, sign consistency, contribution rank per term.
   One-fit terms are not physics. Conjugate-pair terms are grouped, never interpreted independently.
4. Nested candidate formulas (5/10/20-term) per dominant cumulant type, SEPARATELY per V7 branch
   (instantaneous outside, trace-aware inside); no forced universal formula if regimes differ.
5. Rollout is the test: candidates selected on 35/36 in closed loop (all four regions + M=8/10), frozen,
   tested ONCE on 37/38/39. "Candidate closure law" requires stability + most of the relevant V7
   advantage in rollout. 289 nonzero coefficients is a BRIDGE MODEL, not a law.

## FORMULA-SELECTION RULE (numeric, locked before the stability tables arrive)
1. Candidate pool per branch = (top terms by that branch's per-region median contribution) INTERSECT
   (stability-selected: frequency >= 0.8 AND sign consistency >= 0.9 across config resamples), conjugate
   groups intact. BRANCHES SEPARATE: V4-side (instantaneous, Q21 features) selected on normal+seam cells;
   V6-side (trace-aware, Q27 features) selected on memory-box cells. Never pooled.
2. Nested ladder 5 -> 10 -> 20 terms per dominant type-group; weights refit by least squares on the same
   balanced training rows restricted to the selected terms; all non-dominant types closed at zero (the
   original discovery: only these type-groups carry the dynamics).
3. SELECTION (dev seeds 35/36, closed loop): choose the SMALLEST rung that is (a) 100% stable on its
   branch's dev cells and (b) preserves >= 70% of V7's error reduction vs plain 2nd-order, i.e.
   (NRMSE_plain - NRMSE_formula) / (NRMSE_plain - NRMSE_V7) >= 0.70 by branch-region median. The 70%
   share is fixed HERE. If no rung qualifies for a branch: that branch has NO compact formula, and V7 is
   reported for it as a VALIDATED COMPUTATIONAL CLOSURE rather than a discovered compact law.
4. FREEZE after selection; ONE final test on 37/38/39; interpretability tuning stops there permanently.
5. WORDING: trace-term importance is "CONSISTENT WITH a history-dependent correction" -- never proof of a
   formal memory kernel (traces may partly encode burst direction/phase). The mechanism claim rests on
   the dynamical test: whether compact instantaneous formulas fail in the box while trace-bearing ones
   survive.
6. FINAL VERDICT CRITERIA (37/38/39, same teeth as selection): a frozen formula PASSES only with 100%
   stability on its branch cells AND >= 70% advantage share. ONE new instability anywhere in its branch
   = FAIL, regardless of accuracy. "Mostly accurate when it does not explode" is not a closure law.

## INTERPRETABILITY CLOSED (permanent)
Outcome 4 of the pre-registered tree, in its pre-written words (Result 21). No formula frozen; seeds
37/38/39 never consumed (remain clean). No further term mining, ranking changes, or formula attempts.
Active phase: finishing (runtime/amortization, VSC mapping, red team, poster/whiteboard).

## MEMORY-CRITERION EXTENSION (post-project; separately pre-registered, V7 frozen)
TERMINOLOGY: "analytically-motivated, once-calibrated, parameter-only criterion" -- NEVER "a-priori law"
(the threshold is calibrated). Target is PROPENSITY to need memory, not a perfect per-run classifier.
DERIVED number (from the EOM, not fit): D_a = 4 g_c / (kappa + gdown + 2 gphi) = Omega_coll / Gamma_dec,
Omega_coll = 2 g_c (collective Rabi, generates correlations), Gamma_dec = kappa/2 + gdown/2 + gphi
(destroys them). Memory needed when D_a > D_crit. At fixed g_c this is a threshold on s = kappa + 2 gphi.

STEP 0 RESULT (existing data, fixed g_c=0.7, M=5; step0_criterion.py): CONDITIONAL PASS. s=kappa+2gphi
separates memory-needed(V4 fails/V6 repairs) from memoryless-fine at AUC 0.909 (held-out 0.811); both
channels independently confirmed (fixed-gphi kappa-sweep propensity 0.93->0.80->0.50->0.00; w=0 AUC 0.71).
CAVEATS on record: (i) exact 1:2 weight unresolved (flat weight-scan w in [1,10] -- kappa range too
narrow); (ii) single ratio only TIES a dev-fit rectangle on the fixed slice (0.811 vs 0.813), beating only
the deployed box. The criterion's unique value (g_c prediction) is therefore UNTESTED at Step 0.

STEP 1 PRE-REGISTRATION (the decider; requires new runs, V4/V6 frozen):
  - New axis: collective coupling g_c in {0.4, 0.7, 1.0, 1.3} (0.7 is the calibration slice).
  - Also widen kappa at fixed low gphi=0.02 (kappa in {0.02..0.08}) to resolve the 1:2 weight.
  - FROZEN before runs: D_crit calibrated on the g_c=0.7 slice ONLY (one constant); candidate family
    {D_a; D_b=D_a*sqrt(M); D_c=(4g_c^2/kappa)/(kappa/2+gdown/2+gphi)} selected on the g_c=0.7 slice, winner
    frozen. Reference: seeded mcsolve/mesolve at each (g_c, kappa, gphi, seed) to the feasible M ceiling.
  - SUCCESS: the memory-needed boundary at g_c != 0.7 lands on the FROZEN D_crit level set within
    reference uncertainty -- i.e., s_crit(g_c) scales as 4 g_c / D_crit - gdown as predicted; and the
    resolved kappa-weight is consistent with the derived value. Report as propensity (cell-level fraction),
    not hard classification.
  - KILL: boundary at new g_c needs a different D_crit per g_c (no collapse); OR the kappa-weight comes
    out far from derived; OR V4/V6 advantage shows no monotone relation to D_a. Then report the honest
    negative: the failure region is not a single-timescale phenomenon; kappa and gphi gate through
    distinct channels (a data-driven diagnostic, a la arXiv:2512.13913, remains necessary).
  - Seeds: calibrate on the existing seed pool; the g_c!=0.7 runs use fresh seeds never used to set D_crit.

## MEMORY-CRITERION WORDING LOCK (before Step 1 results)
The criterion predicts when the FROZEN memoryless closure (V4) becomes inadequate and the trace-aware V6
is needed -- i.e. the accuracy/stability frontier of the TESTED V4/V6 system. NEVER "predicts when all
instantaneous closures fail": the baseline ladder (Result 20) showed weak instantaneous closures (linear)
stay stable while being ~2x less accurate, so they do not sit on the same frontier. Reporting order:
publish the RAW per-(g_c,kappa,gphi,seed) panel first (V4 exact error+stability, V6 exact error+stability,
s, predicted route, observed label) -- then the summary metrics. FNR (criterion says V4 safe but V4 fails)
is reported explicitly and never hidden under a good AUC.

## V7 FINAL FRAMING LOCK (criterion line CLOSED; see FINDINGS Result 23)
The single-ratio memory criterion is FALSIFIED out-of-distribution and STOPPED. No further criterion work.
Locked wording for the permanent record, poster, abstract, and interviews:

FINAL SCIENTIFIC FRAMING (lead with this):
"We built a validated routed learned closure, mapped its stability-accuracy frontier, showed that finite
memory repairs a specific long-coherence regime, and tested -- but rejected -- a stronger universal
parameter-only routing law."

TREND WORDING (never overstate): "Across the clean reference-supported slices, the observed need for memory
was consistent with increasing collective coupling; however, the strong-coupling slice was partly limited by
V6 out-of-distribution instability and was not used to validate a universal scaling law." NEVER say
"established monotonic memory demand across a 2.5x coupling range."

NON-NEGOTIABLES:
  - Lead the poster/abstract with the validated V7 method, NOT the failed criterion. Result 23 goes in a
    rigor/limitations panel or is used in interviews to prove the stronger hypothesis was tested and killed.
  - The criterion failing does NOT make V7 fail. It means no universal physical rule for choosing V4 vs V6
    is claimed. V7's routing box is an empirically calibrated engineering boundary, not a law.
  - Do NOT adopt a Purcell g_c^2/kappa rate: it does not apply in the tested good-cavity regime (g_c >> kappa)
    and fitting it to the superlinear trend would be post-hoc, not theory.
  - PENDING external theory sanity check (no new model work): an outside quantum-optics person
    independently verifies the sign convention, gperp = gdown/2 + gphi, and the eigenvalue derivation before
    the inverted-state linearization is treated as final physics.

## PHYSICS-PREDICTION EXTENSION PRE-REGISTRATION (superradiant burst vs M) -- written BEFORE any run
GOAL (the ISEF-caliber headline, if it survives): show that the STANDARD 2nd-order cumulant closure's error
on a MEASURABLE collective-emission observable is not random -- it grows systematically with molecule number
M in the long-coherence pocket -- and that the frozen V7 recovers the reference in the validated window.
The certifiable claim is about the STANDARD METHOD's characterized failure; V7 beyond the exact ceiling is
labeled UNVALIDATED extrapolation, never asserted as truth (FINDINGS bottom line already states this).

OBSERVABLES (measurable emitted light = kappa*<a^dag a>(t)):
  - F_peak  = kappa * max_t <a^dag a>(t)         peak emitted flux (superradiant burst height)
  - t_peak  = argmax_t <a^dag a>(t)               burst delay (superradiant timing)
  - I_tot   = kappa * trapz(<a^dag a>, t)         integrated emitted photons (peak-finding-robust check)
  Global max over t in [0,40], NT=200 (unambiguous, reproducible). All three read out of the SAME n(t)
  each method already produces -- no new fit, no new model.

REGIMES (gdown=0.02, sigma=0.15 fixed):
  - POCKET  : gc=0.70, kappa=0.03, gphi=0.02  -> INSIDE V7 memory box (routes to V6). Long-coherence
              regime where plain cumulant is established to fail. Primary test bed.
  - CONTROL : gc=0.70, kappa=0.15, gphi=0.02  -> OUTSIDE box (routes to V4). Negative control: plain
              cumulant should be ~correct here; V7 must not hurt. Establishes regime-SPECIFICITY.

METHODS (all FROZEN, fair): reference (exact mesolve M<=5; mcsolve M=6..<=12, MC error reported) |
  plain 2nd-order cumulant (closure=None, parameter-free) | kappa4=0 T3 truncation (M<=4, its pre-reg cap) |
  V7 (frozen router). Closure engines use the SAME config as all prior V7 evals (build_cell: NC=12, Fsafe=6);
  references use larger Nc with the Fock guard (nc_peak/Nc < 0.45 or the cell is dropped and logged) --
  identical discipline to Step 1.

SEEDS: fresh disorder seeds {50,51,52}, disjoint from train {5,15,30,105,...}, pilot {7}, step0 {19..34},
  step1 {40,41,42}. No seed reuse. Per-M report mean +/- seed spread; for mcsolve also the MC error.

METRICS: relative error of F_peak and I_tot, absolute error of t_peak, for each method vs reference, as a
  function of M. The HEADLINE quantity is the TREND (monotone growth / slope) of the plain-cumulant error
  with M, and whether V7's error stays flat/small over the same range.

KILL RULES (pre-registered; any one kills the headline, reported as a null either way):
  K1. If plain-cumulant F_peak (and t_peak) error does NOT grow with M in the pocket across the validated
      window (flat or non-monotone within reference/MC error) -> NO systematic-failure headline.
  K2. If V7 does not track the reference burst observable better than plain cumulant in the pocket (V7 error
      >= plain error within reference/MC error) -> V7 gives no burst advantage -> NO headline.
  K3. If the negative control fails (plain cumulant is already wrong in the normal regime, OR V7 hurts
      there) -> the "collective-regime-specific" claim is undermined -> reframe, no specificity claim.
CHEAP-FIRST: run EXACT-ONLY M=2..5 (pocket+control, plain/V7/T3) as the kill test BEFORE spending mcsolve.
  If K1/K2 already fail at M<=5, stop and report the null; do not extend to mcsolve.
NO leakage/fitting: V7 frozen, plain and T3 parameter-free, no selection on these outcomes. The extrapolation
  region (M beyond the exact/mc ceiling) is shown only as "best available estimate, exact-UNVALIDATED".

OUTCOME (recorded post-run; see FINDINGS Result 24): K1 TRIGGERED -> scaling headline FALSIFIED. Plain-cumulant
  relative burst-peak error is FLAT ~0.28-0.30 across M=2..5 (both regimes), not growing; mcsolve extension NOT
  run per the pre-registered kill rule. K2 survived (V7 << plain: ~0.005 vs ~0.29). K3: plain overshoots in the
  CONTROL regime too -> NOT pocket-specific (specificity claim dropped). SURVIVING finding: standard 2nd-order
  cumulant OVERSHOOTS the burst peak by a robust +29% (signed), V7 recovers it to +/-0.7% on the same engine/
  cutoff -- a CONSTANT offset, not a scaling law. Structural reason flat: g_c held M-independent. LINE BANKED
  (user decision): no fixed-g retrain, no further observables; push independent replication instead.
  Replication artifact: replicate_cumulant.jl + results/replication_params.json + REPLICATION.md.

## MECHANISM ANALYSIS PRE-REGISTRATION (final frozen internal computation) -- written BEFORE any run
GOAL: is the exact superradiant burst-peak correction driven by ONE dominant connected-third-cumulant (kappa3)
family, a SMALL set, or is it broadly distributed? NO retrain, NO tune, NO reroute, NO new V7 variant. Frozen
V7, frozen engines. Ablations zero a kappa3 family on the CLOSURE OUTPUT only (frozen weights untouched).

TERMINOLOGY (locked): plain 2nd-order sets the CONNECTED kappa3 = 0 (NOT the raw 3rd moment). kappa4=0 T3
EVOLVES kappa3 and closes ITS dynamics through discarded connected 4th-order terms (do NOT say "factorizes
kappa3"). Do NOT assume the photon-polarization-inversion triple is the driver; rank the top families empirically.

DEFINITIONS:
  - kappa3_exact(k,t) = <k>_exact - v3_factorized(m2_exact) (connected cumulant; reuses closure_terms).
    GATE: kappa3_exact(t=0) ~ 0 on the product state (no connected correlations), tol 1e-6.
  - FAMILY = canonical_sig(k)[0] (conjugation-canonical factor-type signature, e.g. (ad,a,sz)=photon-number-
    inversion, (a,sp,sz)/(ad,sm,sz)=photon-polarization-inversion, (a,sp,sm)=photon-spin-exchange, ...). Each
    family is conjugation-CLOSED (asserted) -> zeroing it preserves Hermiticity structurally.
  - DRIVER signal (the "relevant lower-order derivative"): d<a^dag a>/dt has NO direct kappa3 term, so the
    mechanistic target is one level up -- the collective photon-emitter coherence sum_i <a sigma^+_i> whose
    buildup sets the peak (d<a^dag a>/dt emission term ~ -2g sum_i Im<a sigma^+_i>). Family contribution
    E_f(t) = sum_i Im( T3[row(a sigma^+_i)] . (kappa3 masked to family f) ). Cross-check: D_f = ||T3.kappa3_f||.

CELLS/SEEDS (pre-registered): POCKET (gc=0.70,kappa=0.03,gphi=0.02) and CONTROL (gc=0.70,kappa=0.15,gphi=0.02);
  fresh unused seeds 37/38/39 ONLY. Primary molecule count M=4 (largest where T3 is available for the ladder
  comparison; T3 DIVERGES in the pocket at M=4 per Result 24 -> pocket T3 shown at M=3, flagged). M=5 exact/V7/
  plain confirmation that the family ranking + ablation conclusion are M-stable. Burst window = { t : flux
  kappa*<a^dag a>(t) >= 0.5 * exact peak }.

OUTPUTS: (A) kappa3 trajectories exact / V7 / plain(=0) / T3 for the top-3 families in the burst window;
  (B) teacher-forced derivative decomposition: feed EXACT retained <=2 moments (and, for V6, exact-history
  traces) into each closure, decompose E_f by family for exact vs V7; (C) closed-loop ablation of each whole
  family in frozen V7 -> fraction of V7 burst-peak advantage lost = (err_ablate_f - err_v7)/(err_plain - err_v7),
  seed-averaged. One final table + one figure.

PRE-REGISTERED DECISION (choose exactly one; do NOT interpret partial results):
  A family is a MECHANISTIC DRIVER iff ALL: (1) it dominates the teacher-forced exact E_f share in the burst
  window (>=50% of sum_f |E_f| AND >2x the second family); (2) V7 reproduces its sign, in-burst timing, and
  magnitude (ratio in [0.5,2]); (3) its ablation removes a substantial share of V7's burst-peak advantage
  (frac_lost >= 0.5), consistently across seeds 37/38/39.
  -> Conclusion 1 (single-family) if exactly one family meets all three.
  -> Conclusion 2 (small-family distributed) if 2-3 families TOGETHER meet the bar but no single family does.
  -> Conclusion 3 (broadly distributed, no compact attribution) otherwise.
  After this: FREEZE all modeling; move to external replication, manuscript, ISEF poster, whiteboard defense.

OUTCOME (recorded post-run; see FINDINGS Result 25): CONCLUSION 2 = SMALL-FAMILY DISTRIBUTED. Two connected
  kappa3 families carry ~100% of the burst-peak correction (all others ~0), robust across seeds 37/38/39 and
  both regimes: DOMINANT photon-number-inversion <a^dag a sigma^z> (D-share ~0.66; ablation removes ~0.82;
  critical in pocket), SECONDARY photon-polarization-inversion <a sigma^+ sigma^z> (D-share ~0.34; ablation
  ~0.26). NOT the triple alone (that is the secondary). V7 reconstructs both (sign+magnitude); plain(kappa3=0)
  misses the large negative dominant cumulant -> +29% overshoot; kappa4=0 T3 mis-estimates/diverges -> -28%
  undershoot. GATE kappa3_exact(0)<1e-6 passed all cells. Figure results/mechanism_fig.png. The pre-registered
  DIRECT driver E_f is structurally single-family (disclosed); D_f + ablation are the non-circular adjudicators.
  MODELING FROZEN. Scripts: mechanism.py (compute), mechanism_report.py (decision+figure).

## DISORDER-SWEEP KILL TEST (candidate Phase II) -- pre-registered in-session, recorded here
HYPOTHESIS (candidate, NOT a discovery): does spectral disorder sigma produce a non-trivial (ideally
non-monotonic, intermediate-sigma) change in the two Result-25 connected-third-order channels AND make the
2nd-order closure meaningfully more wrong, tied to a burst-observable change? Exact + plain 2nd-order only
(NO V7), fixed g_c (=> G^2 fixed), kappa=0.03, gphi=0.02, full inversion, M=3,4, held-out seeds 60-64,
sigma in {0,0.05,0.10,0.15,0.20,0.30,0.45}. SURVIVE = non-monotonic structure in a kappa3 channel AND the
2nd-order error at signal/noise>3, tied to F_peak/t_peak. KILL = monotone / noise-level.
OUTCOME (see FINDINGS Result 26): KILL. Every RESOLVED quantity (signal/noise>3) is MONOTONE; the only
non-monotonic flag (kappa3 total, M=4) is noise (s/n 1.7). Disorder monotonically: suppresses F_peak (~16%),
leaves t_peak flat (unresolved), and REDISTRIBUTES kappa3 from photon-number-inversion (down ~15-20%) to
photon-polarization-inversion (up +66-83%, s/n up to 12.8) at ~constant total, while the 2nd-order error rises
only ~3 points (28->31%). No new failure regime; standard-method failure is ~disorder-robust at fixed G. The
candidate is dead as novelty. Scripts: disorder_sweep.py, disorder_report.py; figure results/disorder_sweep_fig.png.
Third killed extension (after Results 22-23 criterion and Result 24 M-scaling): extension well is dry.

## FOURTH-ORDER CUMULANT BASELINE (the fairness test) -- pre-registered, tightly scoped
QUESTION (the one baseline that can challenge the core "learned closure is useful/necessary" claim): does a
PROPERLY IMPLEMENTED fourth-order cumulant closure (track moments to order 4, set connected kappa5=0) solve the
disordered open-cavity burst peak at M=3,4 -- and at what COST and STABILITY? NOT "build a giant engine and hope
it breaks": build it correctly, GATE it, and read the result. Note (correcting the benchmark paper): its "4th
order exact at N=4" is spin-only free-space Dicke (finite spin space closes at order N); OUR model has a cavity
Fock mode + loss + dephasing + disorder, so 4th order does NOT trivially become exact -- must be tested.

BUILD/CORRECTNESS: T4Engine mirrors the validated T3 (kappa4=0) engine one order up. kappa5=0 closure of each
order-5 moment via the general set-partition moment-cumulant formula m_n = sum_{partitions |pi|>=2} (-1)^|pi|
(|pi|-1)! prod m(block) (verified to reproduce T3's kappa4 terms exactly at n=4). UNFORGEABLE GATE (required
<1e-8 before ANY rollout is trusted): feeding ALL-exact moments, assembled d<k>/dt == exact <L^dagger[k]>.
Same Fock-boundary-masked least-squares construction as the order-2 and order-3 engines.

SCOPE: M=3 primary (gate + accuracy), M=4 attempted and its COST reported (O(M^5) tracked set -- cost is part of
the answer). Cells POCKET (gc=0.7,kappa=0.03,gphi=0.02) and CONTROL (gc=0.7,kappa=0.15,gphi=0.02); seeds
50-52 (SAME as Result 24 burst_pred, so plain/V7 numbers are directly comparable). Metrics: signed F_peak error
vs exact, STABILITY (blows up in the pocket?), and COST (tracked-set size, build time, rollout time).

PRE-REGISTERED OUTCOMES (choose from the exact data; no spin):
  O1. T4 gets the peak right AND is practical -> weakens "V7 necessary"; reframe V7 as a possible ALTERNATIVE
      (maybe cheaper), not a necessity. Report honestly.
  O2. T4 accurate but expensive/unstable in the pocket -> favorable for V7 (comparable accuracy, lower cost,
      survives the hard regime).
  O3. T4 still fails / blows up -> strongest for V7 (going up the standard hierarchy gives no practical fix here).
DO NOT overclaim: this settles the fairness question (is V7 a meaningful practical closure vs a fancy substitute
for a known method); it does NOT make the project field-defining. Scripts: t4_truncation.py, t4_eval.py.

OUTCOME (see FINDINGS Result 27): O3 (strongest for V7). Engine correctly built (closure formula == T3 kappa4
terms; unforgeable gate PASSES at ~1e-14). The kappa5=0 fourth-order truncation GENUINELY DIVERGES (finite-time,
t~9.65) at M=3 in BOTH regimes -- confirmed NOT a numerics artifact: 4 solvers (LSODA rtol 1e-5 AND 1e-8, plus
stiff implicit Radau and BDF) all diverge at the IDENTICAL time and magnitude (8.8e5). It also overshoots the
peak before diverging, and is expensive/erratic even at M=3 (build 72-1068s across draws). So among the TESTED
zero-next-cumulant truncations through 4th order, NONE gave a stable, accurate burst closure in the EVALUATED
cells: 2nd overshoots (+29%), kappa4=0 3rd undershoots (-28%, runaway growth triggering the guard in the pocket),
kappa5=0 4th shows runaway growth triggering the guard in all evaluated cells (held-out draws 50-52); V7 is
+/-0.7%, stable, and avoids the high-order engine construction. (Do NOT claim no such method could
exist; only the tested truncations failed in the evaluated cells.) Likely cause of divergence (differs from spin-only
Dicke benchmark where 4th order helps): the unbounded cavity Fock mode, whose moment hierarchy is prone to
finite-time truncation divergence. Fairness question settled in V7's favor; NOT field-defining.
