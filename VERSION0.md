# VERSION 0 FREEZE (control group for all future changes)

Git commit: `ea753840a623ffd5e547c2b35c5fafae64685042` (tagged `v0`)
Date frozen: 2026-07-01

## Environment
- Apple M4, 10 cores, 16 GB RAM, macOS (Darwin 25.4.0)
- Python 3.12.13, NumPy 2.4.6, SciPy 1.17.1, scikit-learn 1.9.0, QuTiP 5.3.0
- venv: `~/polariton-research/sim/golden/.venv`

## Solver settings (all V0 results)
- Exact reference: `qutip.mesolve` defaults (adaptive; atol 1e-8 / rtol 1e-6); trajectories: `qutip.mcsolve`,
  ntraj=200, seeds=12345 (deterministic), MC error reported per run
- Cumulant/closure loop: `scipy.solve_ivp` RK45, rtol=1e-4, atol=1e-6, NT=200 points over TMAX=40
- Model: WC=1.0, g_i = 0.8/sqrt(M), KAPPA=0.1, GDOWN=0.02, GPHI=0.02, GPUMP=0; Gaussian frequency disorder;
  fully-inverted product initial state, cavity vacuum
- Engines: dense CumulantEngine (Fsafe=6, gate ~1e-14), ScalableEngine (== dense C at M=2..5 to ~1e-14)

## Frozen splits
- v2 closure training: M in {2,3}, seeds 0-5, sigma in {0.05,0.10,0.15,0.20,0.30}  (transfer_train.npz)
- "retrained" closure: + M in {4,5}, seeds 0-3, sigma in {0.05,0.15,0.30}          (transfer_train_v2.npz)
- Test: disorder seed 7 (never trained), sigma 0.15; test M in {4,5,8,10} never in the respective training set
- Scalers fit on training data only (inside LocalClosure.fit)

## Headline V0 metrics (max-abs error over t<=40 in {n, mean sz}, held-out seed 7)
| M | ref | plain 2nd-order | closure(2,3) | closure(2..5) |
|---|---|---|---|---|
| 4 | mesolve | 1.053 | 0.414 | -- |
| 5 | mesolve | 1.296 | 0.469 | 0.348 |
| 8 | mcsolve (MCerr 0.096) | 1.967 | 1.307 | 0.856 |
| 10 | mcsolve (MCerr 0.119) | 2.503 | 2.398 | 0.890 |
- M=12/16/20 with closure(2..5): stable rollouts, physically-shaped, UNVALIDATED beyond M=10 (numerical
  reach, not accuracy). z_end undershoots -1 by 0.4-1.3% at M>=12.
- In-distribution M=3 held-out realization: 0.023 (37x better than plain).

## Frozen artifacts (SHA-256, first 16 hex)
- transfer_train.npz     befe0851c848642d
- transfer_train_v2.npz  fab5ce243d4fbdaa
- lc_old.pkl             c0e3805b44eda2f2
- lc_new.pkl             bbb4f9d0b33e0e35
- retrain_eval.npz       4cfdc9ca32f0944c
- transfer_trend.npz     a7039f9183ed2904

Architecture/hyperparameters: local permutation-equivariant closure, one MLPRegressor(128,128), alpha=1e-5,
early_stopping, n_iter_no_change=25, random_state=0 per cumulant type; features width 18
(singles/pairs re+im, site detunings, context [nc, mean sz, mean |coh|]); kappa3 clipped to training range.

Any future claim of improvement is measured against THIS table with THESE splits and settings.
