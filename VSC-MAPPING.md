# Physical parameter mapping: the swept model regimes vs one real platform

Platform chosen (single, concrete, per red-team #11 and finishing item 3): **vibrational strong coupling
(VSC) of a molecular vibration in a Fabry-Perot microcavity** -- e.g. a C=O or C-H stretch of an organic
liquid/film tuned to a mid-IR cavity mode (the Ebbesen-type VSC systems). Single-mode, collective coupling
of many identical-vibration molecules with inhomogeneous frequency broadening: the closest real analogue
of this model's structure (single mode + collective + frequency disorder + local dephasing).

This is a MAPPING, not a claim of quantitative realism (see FINDINGS SCOPE #1). It shows which model axes
correspond to which physical knobs, and honestly where the correspondence breaks.

## Representative VSC numbers (order of magnitude, mid-IR vibration)
- Vibration frequency omega_v ~ 1700 cm^-1 (C=O stretch) ~ 5.1e13 Hz -> this is our wc = 1 (units).
- Collective Rabi splitting Omega_R (observed): ~ 20-200 cm^-1 in strong VSC -> Omega_R/omega_v ~ 0.01-0.12.
- Cavity linewidth kappa: mid-IR FP cavities Q ~ 10-100 -> kappa/omega_v ~ 0.01-0.1.
- Vibrational dephasing gamma_phi (T2*): ~ 1-10 ps -> gamma_phi/omega_v ~ 1e-3 - 1e-2.
- Vibrational relaxation gamma_down (T1): ~ 1-50 ps -> gamma_down/omega_v ~ 2e-4 - 1e-2.
- Inhomogeneous broadening sigma: ~ 5-20 cm^-1 -> sigma/omega_v ~ 0.003-0.012.

## Model axis -> physical knob, and where it maps
| model | physical meaning | model range (this project) | realistic VSC | verdict |
|---|---|---|---|---|
| g_coll (Rabi/2, collective) | light-matter coupling | 0.4-1.1 wc | 0.005-0.06 wc | MODEL EXAGGERATED ~10-40x: our couplings probe robustness, not VSC magnitudes |
| kappa | cavity loss | 0.03-0.4 wc | 0.01-0.1 wc | OVERLAPS at the low end; our high-kappa cells are beyond typical VSC |
| gphi | vibrational dephasing (pure) | 0.02-0.3 wc | 0.001-0.01 wc | MODEL EXAGGERATED ~2-30x (chosen so decoherence fits t<=40) |
| gdown | vibrational relaxation | 0.02 wc fixed | 0.0002-0.01 wc | MODEL EXAGGERATED, and un-swept (SCOPE #2) |
| sigma | inhomogeneous broadening | 0.05-0.4 | 0.003-0.012 | MODEL EXAGGERATED ~4-30x |
| M | molecules per mode | 2-20 (validated), rollout to 20+ | 10^6-10^10 | METHOD gap is the point: exact stops ~7, real M is astronomical |

## The one honest sentence
The model's DIMENSIONLESS ORDERINGS (which rates are larger/smaller, and the low-decoherence corner where
memory matters) are the transferable content; the absolute rates are compressed ~1-2 orders of magnitude
above real VSC so every dynamical regime fits a t<=40 window. The genuinely realistic axis is M: real VSC
has 10^6-10^10 molecules per mode, exact simulation dies at ~7, and a transferable reduced closure is
exactly the tool that gap calls for. Quantitative VSC prediction would require: real rate ratios (longer
horizons), a spectral (two-time) observable via quantum regression, and coupling/multimode disorder --
all declared future work.

## What this buys the project
It converts "this might help quantum optics" into: "this method targets the specific, real regime of
disordered collective vibrational strong coupling, where the molecule count that makes exact simulation
impossible is precisely the regime a transferable closure addresses -- with the honest caveat that the
present rates are model-scale, not VSC-quantitative."
