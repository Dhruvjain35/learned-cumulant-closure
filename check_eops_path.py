"""
Equivalence check for the new training-data path: at M=3 (where the dense engine still exists), the moments
returned by mesolve e_ops (gen_data_v2.moments_by_eops) must equal the dense engine's moments_from_rho on the
same trajectory, for every order<=3 monomial at every sampled time. If they match, the new path's kappa3 and
features are identical-by-construction (same downstream code on the same numbers).
"""
import numpy as np
import qutip as qt
from cumulant import CumulantEngine
from gen_data_v2 import moments_by_eops, WC, GCOLL, KAPPA, GDOWN, GPHI, TMAX, NT
from exact_tc import draw_disorder

M, Nc, sigma, seed = 3, 10, 0.15, 0

w = draw_disorder(M, WC, sigma, seed, "gauss")
g = np.full(M, GCOLL / np.sqrt(M))

# new path
keys, moms, nc_peak = moments_by_eops(M, Nc, w, g)

# old path: mesolve states -> dense engine moments_from_rho
eng = CumulantEngine(M, Nc, WC, w, g, kappa=KAPPA, gdown=GDOWN, gphi=GPHI, Fsafe=6)
psi0 = qt.tensor(qt.basis(Nc, 0), *[qt.basis(2, 0) for _ in range(M)])
ts = np.linspace(0, TMAX, NT)
states = qt.mesolve(eng.H, psi0, ts, c_ops=eng.cops, options={"progress_bar": False}).states

worst = 0.0
for ti in range(0, NT, 20):
    mfull = eng.moments_from_rho(states[ti])
    for k in keys:
        d = abs(moms[k][ti] - mfull[eng.idx[k]])
        worst = max(worst, d)
print(f"max |moments(e_ops) - moments(dense rho)| over {len(keys)} keys x sampled t = {worst:.2e}")
print("PASS" if worst < 1e-6 else "FAIL")
