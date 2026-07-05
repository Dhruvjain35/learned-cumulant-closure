"""
Exact open Tavis-Cummings solver with per-emitter (frequency) disorder, using QuTiP 5.3.

Model (RWA, cavity-first tensor convention to match the validated simulator):
    H = wc a^dag a + sum_i w_i sp_i sm_i + sum_i g_i (a sp_i + a^dag sm_i)
Open dynamics (Lindblad):
    drho/dt = -i[H,rho] + kappa D[a] + sum_i gpump D[sp_i] + sum_i gphi/2 D[sz_i] + sum_i gdown D[sm_i]
    D[O]rho = O rho O^dag - 1/2 {O^dag O, rho}

We measure MEAN-FIELD (one-point) quantities and the BEYOND-MEAN-FIELD connected
two-point correlations, which are exactly what a mean-field factorization drops:
    chi_i  = <a^dag sm_i> - <a^dag><sm_i>            (cavity-emitter connected)
    C_ij   = <sp_i sm_j> - <sp_i><sm_j>   (i != j)   (emitter-emitter connected)
If these are ~0, mean-field is essentially exact and there is nothing to learn.
If they are substantial and structured, mean-field fails by that amount.

Everything here is exact (full Lindblad); nothing is approximated. Validated against
known limits in validate_model.py before any physics claim is made.
"""
import numpy as np
import qutip as qt


def build_ops(M, Nc):
    """Cavity-first operators on the (Nc x 2^M) space. Returns dict of QuTiP operators."""
    ids = [qt.qeye(2)] * M
    a = qt.tensor(qt.destroy(Nc), *ids)
    sm, sp, sz, ee = [], [], [], []
    for i in range(M):
        lst = [qt.qeye(Nc)] + [qt.qeye(2)] * M
        lst[i + 1] = qt.sigmam()
        smi = qt.tensor(lst)
        sm.append(smi)
        sp.append(smi.dag())
        sz.append(sp[-1] * smi - smi * sp[-1])   # sz = sp sm - sm sp = |e><e| - |g><g|
        ee.append(sp[-1] * smi)                   # excited-state projector |e><e|
    return {"a": a, "sm": sm, "sp": sp, "sz": sz, "ee": ee, "M": M, "Nc": Nc}


def build_H(ops, wc, w, g):
    """H with per-emitter frequency w[i] and coupling g[i] (arrays of length M)."""
    a, sm, sp, ee = ops["a"], ops["sm"], ops["sp"], ops["ee"]
    M = ops["M"]
    H = wc * a.dag() * a
    for i in range(M):
        H += w[i] * ee[i]
        H += g[i] * (a * sp[i] + a.dag() * sm[i])
    return H


def collapse_ops(ops, kappa=0.0, gpump=0.0, gphi=0.0, gdown=0.0):
    """Cavity loss kappa, incoherent pump gpump (sp), pure dephasing gphi (sz), decay gdown (sm)."""
    a, sm, sp, sz = ops["a"], ops["sm"], ops["sp"], ops["sz"]
    M = ops["M"]
    c = []
    if kappa > 0:
        c.append(np.sqrt(kappa) * a)
    for i in range(M):
        if gpump > 0:
            c.append(np.sqrt(gpump) * sp[i])
        if gdown > 0:
            c.append(np.sqrt(gdown) * sm[i])
        if gphi > 0:
            c.append(np.sqrt(gphi / 2.0) * sz[i])   # gives (gphi/2) D[sz] -> coherence decay rate gphi
    return c


def draw_disorder(M, wc, sigma, seed, family="gauss"):
    """Per-emitter frequencies centered at wc with inhomogeneous width sigma."""
    rng = np.random.default_rng(seed)
    if sigma <= 0:
        return np.full(M, wc)
    if family == "gauss":
        return wc + rng.normal(0.0, sigma, M)
    if family == "lorentz":
        return wc + sigma * rng.standard_cauchy(M)
    if family == "uniform":
        return wc + rng.uniform(-np.sqrt(3) * sigma, np.sqrt(3) * sigma, M)   # matched std
    if family == "bimodal":
        s = rng.integers(0, 2, M) * 2 - 1
        return wc + s * sigma
    raise ValueError(family)


def measure(rho, ops):
    """One-point moments and connected two-point correlations from the exact state rho."""
    a, sm, sp, ee = ops["a"], ops["sm"], ops["sp"], ops["ee"]
    M = ops["M"]
    nc = qt.expect(a.dag() * a, rho).real
    alpha = qt.expect(a, rho)
    adag = np.conj(alpha)
    pe = np.array([qt.expect(ee[i], rho).real for i in range(M)])          # excited populations
    smv = np.array([qt.expect(sm[i], rho) for i in range(M)])              # <sm_i>

    # cavity-emitter connected: chi_i = <a^dag sm_i> - <a^dag><sm_i>
    chi = np.array([qt.expect(a.dag() * sm[i], rho) - adag * smv[i] for i in range(M)])

    # emitter-emitter connected: C_ij = <sp_i sm_j> - <sp_i><sm_j>, i != j
    C = np.zeros((M, M), dtype=complex)
    Gfull = np.zeros((M, M), dtype=complex)
    for i in range(M):
        for j in range(M):
            gij = qt.expect(sp[i] * sm[j], rho)     # note: different sites commute; i==j gives pe
            Gfull[i, j] = gij
            C[i, j] = gij - np.conj(smv[i]) * smv[j]
    return {"nc": nc, "alpha": alpha, "pe": pe, "smv": smv, "chi": chi, "C": C, "Gfull": Gfull}


def bmf_indices(m):
    """Scalar beyond-mean-field indices from a measure() dict. Dimensionless where possible."""
    M = m["pe"].shape[0]
    pe_mean = max(m["pe"].mean(), 1e-12)
    # emitter-emitter: mean |connected off-diagonal| normalized by mean excited population
    offdiag = [abs(m["C"][i, j]) for i in range(M) for j in range(M) if i != j]
    C_ee = (np.mean(offdiag) / pe_mean) if offdiag else 0.0
    # cavity-emitter connected magnitude, normalized by sqrt(nc * pe) scale
    scale_ce = np.sqrt(max(m["nc"], 1e-12) * pe_mean)
    C_ce = float(np.mean(np.abs(m["chi"])) / scale_ce)
    # fraction of the emitter coherence matrix (off-diagonal) that is connected (= all of it if <sm>=0)
    off_full = np.array([abs(m["Gfull"][i, j]) for i in range(M) for j in range(M) if i != j])
    off_conn = np.array([abs(m["C"][i, j]) for i in range(M) for j in range(M) if i != j])
    frac_conn = float(off_conn.sum() / off_full.sum()) if off_full.sum() > 1e-14 else 0.0
    return {"C_ee_over_pe": float(C_ee), "C_ce_norm": C_ce, "frac_offdiag_connected": frac_conn,
            "mean_abs_Cij": float(np.mean(offdiag)) if offdiag else 0.0,
            "mean_pe": float(m["pe"].mean()), "nc": float(m["nc"])}


def steady(M, Nc, wc, g0, sigma, seed, kappa, gpump, gphi=0.0, gdown=0.0, family="gauss"):
    """Build model, solve exact steady state, return (measure dict, ops, H, c_ops)."""
    ops = build_ops(M, Nc)
    w = draw_disorder(M, wc, sigma, seed, family)
    g = np.full(M, g0)
    H = build_H(ops, wc, w, g)
    c = collapse_ops(ops, kappa=kappa, gpump=gpump, gphi=gphi, gdown=gdown)
    rho = qt.steadystate(H, c)
    return measure(rho, ops), ops, w


if __name__ == "__main__":
    # smoke test: build M=2 and print dims
    ops = build_ops(2, 3)
    print("dims:", ops["a"].dims, "shape:", ops["a"].shape)
