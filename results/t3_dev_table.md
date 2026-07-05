# T3 (kappa4 = 0 third-order truncation) — dev table artifact

Command: `python t3_eval.py` (seeds 29/31). Engine: dense T3Engine (t3_truncation.py), track = all
normal-ordered monomials order <= 3 (pqmax 3), full basis order <= 4 (pqmax 4), Fock-boundary-masked
least-squares projection, Fsafe = 6, Nc = {M=3: 10, M=4: 11}. Closure: standard kappa4 = 0 factorization
over ordered fundamental factors. Rollout: RK45 rtol 1e-4 / atol 1e-6, t in [0, 40], 200 points
(identical to every other closure). Gate enforced per engine build: all-exact moments -> assembled
derivative must equal exact <L†[k]>; every build passed at 5.5e-15 .. 1.9e-14. Hardware: Apple M4.

| M | cell | seed | gate | T3 NRMSE | tracked | build s | roll s | plain-2nd | V7 |
|---|---|---|---|---|---|---|---|---|---|
| 3 | deep (0.030, 0.020) | 29 | 1.7e-14 | 0.377 | 172 | 49 | 2.6 | 0.598 | 0.149 [v6] |
| 3 | deep (0.030, 0.020) | 31 | 1.9e-14 | 0.380 | 172 | 91 | 2.7 | 0.596 | 0.156 [v6] |
| 3 | normal (0.150, 0.020) | 29 | 1.4e-14 | 0.432 | 172 | 23 | 1.1 | 0.447 | 0.049 [v4] |
| 3 | normal (0.150, 0.020) | 31 | 1.5e-14 | 0.431 | 172 | 43 | 2.7 | 0.446 | 0.047 [v4] |
| 4 | deep (0.030, 0.020) | 29 | 5.5e-15 | 0.372 | 352 | 7273* | 2.6 | 0.581 | 0.080 [v6] |
| 4 | deep (0.030, 0.020) | 31 | 6.9e-15 | 0.374 | 352 | 885 | 2.1 | 0.609 | 0.081 [v6] |
| 4 | normal (0.150, 0.020) | 29 | 1.1e-14 | 0.493 | 352 | 616 | 2.6 | 0.422 | 0.039 [v4] |
| 4 | normal (0.150, 0.020) | 31 | 9.4e-15 | 0.488 | 352 | 634 | 3.8 | 0.426 | 0.041 [v4] |

*7273 s build ran concurrently with the disk-full event and other jobs; 616-885 s is the representative
M=4 build time. Order-2 tracked counts for comparison: 60 / 96 / 141 at M = 3 / 4 / 5.

PRECISE CLAIM (wording locked): in the TESTED kappa4=0 third-order truncation, explicitly tracking
third-order moments was substantially more expensive (3.7x state variables at M=4, minutes-scale dense
builds) and still much less accurate than V7 across the tested normal and long-coherence cells. It helps
vs plain second-order in the deep corner (0.60 -> 0.38) but remains 2.5-4.7x behind V7 there, and gives
no benefit (or is worse) in normal cells. This is ONE specific conventional closure -- no claim that
every higher-order cumulant method fails. Gates passed at machine precision: not an implementation excuse.
