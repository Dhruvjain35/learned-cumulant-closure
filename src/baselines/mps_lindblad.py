"""
Matrix-product-state baseline for the open, disordered Tavis-Cummings model.

WHY THIS EXISTS
---------------
Every closure in this repository is compared against other cumulant closures. A referee will
immediately ask why one would not simply run a tensor-network method instead. This module answers
that question with numbers rather than a sentence in the limitations.

METHOD
------
The Lindblad equation is a linear ODE on the density matrix, so we vectorise it and evolve the
resulting state vector as an MPS. With the column-stacking convention

    vec(A rho B) = (B^T kron A) vec(rho),          vec index = col * d + row

a local operator O on a site of physical dimension d becomes two superoperators on a "doubled"
site of dimension d^2:

    O_L = kron(I_d, O)      (left multiplication, O rho)
    O_R = kron(O.T, I_d)    (right multiplication, rho O)

The Liouvillian is then

    L = -i (H_L - H_R) + sum_mu [ kron(conj(L_mu), L_mu) - 1/2 (Ld L)_L - 1/2 (Ld L)_R ]

Every dissipator in this model (cavity loss, emitter decay, dephasing) is single-site in the
doubled space. Only the cavity-emitter coupling is two-site, and because the cavity sits at one
end of the chain those terms are long-range. That rules out TEBD and rules out TDVP (whose Krylov
step assumes a Hermitian generator, which a Liouvillian is not). We therefore use the W_II MPO
approximation of exp(L dt) from Zaletel et al., PRB 91, 165112 (2015), as implemented in TeNPy's
ExpMPOEvolution -- the standard choice for a non-Hermitian generator with long-range terms.

The model solved here is byte-for-byte the model in src/engines/exact_tc.py: cavity-first ordering,
H = wc a^dag a + sum_i w_i |e><e|_i + sum_i g_i (a sp_i + a^dag sm_i), jump operators
sqrt(kappa) a, sqrt(gdown) sm_i, sqrt(gphi/2) sz_i, and sz = |e><e| - |g><g|. Nothing is
re-derived here; the operators are built to match that file exactly so the comparison is fair.

NOTHING IS TRUSTED UNTIL IT PASSES THE GATE. validate_mps.py checks this solver against QuTiP's
exact mesolve before any number produced here is used in a claim.
"""
import time

import numpy as np

from tenpy.networks.site import Site
from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPOGraph
from tenpy.linalg.np_conserved import Array
from tenpy.linalg.charges import LegCharge, ChargeInfo


# --------------------------------------------------------------------------------------
# physical (undoubled) operators, matched to src/engines/exact_tc.py
# --------------------------------------------------------------------------------------

def cavity_ops(Nc):
    """Bosonic annihilation / number on a Fock space truncated at Nc levels."""
    a = np.diag(np.sqrt(np.arange(1, Nc)), 1)
    return {"a": a, "ad": a.conj().T, "n": a.conj().T @ a, "I": np.eye(Nc)}


def emitter_ops():
    """Two-level operators in QuTiP's convention: basis(2,0) is the EXCITED state.

    qutip.sigmam() = [[0,0],[1,0]], so sm |0> = |1>: index 0 is excited, index 1 is ground.
    Then ee = sp sm = diag(1,0) and sz = sp sm - sm sp = diag(1,-1), matching exact_tc.build_ops.
    """
    sm = np.array([[0.0, 0.0], [1.0, 0.0]])
    sp = sm.conj().T
    ee = sp @ sm
    sz = sp @ sm - sm @ sp
    return {"sm": sm, "sp": sp, "ee": ee, "sz": sz, "I": np.eye(2)}


# --------------------------------------------------------------------------------------
# vectorisation helpers
# --------------------------------------------------------------------------------------

def op_L(O, d):
    """Left multiplication O rho, in the column-stacking convention."""
    return np.kron(np.eye(d), O)


def op_R(O, d):
    """Right multiplication rho O, in the column-stacking convention."""
    return np.kron(O.T, np.eye(d))


def dissipator_super(L, d):
    """Superoperator for D[L] rho = L rho Ld - 1/2 {Ld L, rho}, on one doubled site."""
    LdL = L.conj().T @ L
    return np.kron(L.conj(), L) - 0.5 * op_L(LdL, d) - 0.5 * op_R(LdL, d)


def doubled_site(ops, d, label):
    """A TeNPy Site of dimension d^2 carrying the superoperators we need.

    No charge conservation: the Liouvillian does not conserve the obvious quantum numbers once
    dissipation is included, and the chain is short enough that we do not need the speedup.
    """
    leg = LegCharge.from_trivial(d * d, ChargeInfo([], []))
    named = {}
    for name, O in ops.items():
        named[name + "_L"] = op_L(O, d)
        named[name + "_R"] = op_R(O, d)
    site = Site(leg, [str(i) for i in range(d * d)], sort_charge=False, **named)
    site.label = label
    site.phys_dim_undoubled = d
    return site


# --------------------------------------------------------------------------------------
# Liouvillian MPO
# --------------------------------------------------------------------------------------

def build_liouvillian(M, Nc, wc, w, g, kappa, gdown, gphi, scale=1.0):
    """Liouvillian of the open disordered Tavis-Cummings model as a TeNPy MPO.

    Chain ordering is cavity-first: site 0 is the cavity, sites 1..M are the emitters, matching
    the tensor convention of the exact solver.

    `scale` multiplies the whole generator. TeNPy's ExpMPOEvolution builds exp(-i dt H_MPO), so
    obtaining exp(L dt) requires handing it H_MPO = i L. The caller passes scale=1j for that;
    the default of 1.0 returns the physical Liouvillian, which is what the validation gate checks.
    """
    cav, emi = cavity_ops(Nc), emitter_ops()
    sites = [doubled_site(cav, Nc, "cav")] + [doubled_site(emi, 2, "emi") for _ in range(M)]

    # onsite: -i(H_L - H_R) + dissipators
    onsite = []
    Lc = -1j * (wc * op_L(cav["n"], Nc) - wc * op_R(cav["n"], Nc))
    if kappa > 0:
        Lc = Lc + dissipator_super(np.sqrt(kappa) * cav["a"], Nc)
    onsite.append(Lc)

    for i in range(M):
        Li = -1j * (w[i] * op_L(emi["ee"], 2) - w[i] * op_R(emi["ee"], 2))
        if gdown > 0:
            Li = Li + dissipator_super(np.sqrt(gdown) * emi["sm"], 2)
        if gphi > 0:
            Li = Li + dissipator_super(np.sqrt(gphi / 2.0) * emi["sz"], 2)
        onsite.append(Li)

    # two-site cavity<->emitter, from -i(H_L - H_R) with H_coup = g_i (a sp_i + ad sm_i).
    # Operators on distinct sites factorise, so (AB)_L = A_L B_L and likewise for _R.
    couplings = []
    for i in range(M):
        gi = g[i]
        for cop, eop in (("a", "sp"), ("ad", "sm")):
            couplings.append((-1j * gi, 0, i + 1, cop + "_L", eop + "_L"))
            couplings.append((+1j * gi, 0, i + 1, cop + "_R", eop + "_R"))

    graph = MPOGraph(sites, bc="finite", max_range=M + 1)
    graph.add_missing_IdL_IdR()
    for j, Lloc in enumerate(onsite):
        sites[j].add_op("Lonsite", scale * Lloc, need_JW=False, hc=False)
        graph.add(j, "IdL", "IdR", "Lonsite", 1.0)
    couplings = [(scale * s, i, j, oi, oj) for (s, i, j, oi, oj) in couplings]
    for strength, i, j, opi, opj in couplings:
        label = f"c{i}_{j}_{opi}"
        graph.add(i, "IdL", label, opi, strength)
        for k in range(i + 1, j):
            graph.add(k, label, label, "Id", 1.0)
        graph.add(j, label, "IdR", opj, 1.0)
    graph.add_missing_IdL_IdR()
    return sites, graph.build_MPO()


# --------------------------------------------------------------------------------------
# states
# --------------------------------------------------------------------------------------

def _vec_product_mps(sites, local_rhos):
    """Product MPS whose site tensors are vec(rho_local), column-stacking."""
    p_state = [rho.flatten(order="F") for rho in local_rhos]
    return MPS.from_product_state(sites, p_state, bc="finite", dtype=complex)


def initial_state(sites, M, Nc):
    """Cavity vacuum, every emitter excited -- the fully inverted start."""
    rho_c = np.zeros((Nc, Nc), complex)
    rho_c[0, 0] = 1.0
    rho_e = np.zeros((2, 2), complex)
    rho_e[0, 0] = 1.0                        # index 0 is EXCITED in this convention
    return _vec_product_mps(sites, [rho_c] + [rho_e] * M)


def trace_state(sites, M, Nc):
    """vec(identity): contracting against it takes the trace, so <O> = <<I| O_L |rho>>."""
    return _vec_product_mps(sites, [np.eye(Nc, dtype=complex)] + [np.eye(2, dtype=complex)] * M)


# --------------------------------------------------------------------------------------
# measurement and evolution
# --------------------------------------------------------------------------------------

def measurement_bra(sites, M, Nc, site_index, O):
    """Product MPS whose conjugating overlap with |rho>> gives Tr(O rho), for local O.

    Derivation: <<I| O_L = ( (O^dag)_L |I>> )^dag and (O^dag)_L|I>> = vec(O^dag). TeNPy's
    `overlap` conjugates the bra, and conj(vec(O^dag))[c*d+r] = O[c,r], so contracting against
    vec(O^dag) yields sum_{r,c} O[c,r] rho[r,c] = Tr(O rho) exactly.

    Building the measurement as a bra avoids applying an operator to the state, which matters
    because a^dag a annihilates the initial vacuum and TeNPy refuses zero-norm results.
    """
    locals_ = [np.eye(Nc, dtype=complex)] + [np.eye(2, dtype=complex) for _ in range(M)]
    locals_[site_index] = O.conj().T
    return _vec_product_mps(sites, locals_)


def trace_of(rho_mps, tr_mps):
    """Tr(rho). Should stay 1; deviation is a direct measure of truncation damage."""
    return tr_mps.overlap(rho_mps)


class _Lat:
    """Minimal stand-in for a TeNPy lattice: the evolution engine only asks for the site list."""

    def __init__(self, sites):
        self._sites = sites

    def mps_sites(self):
        return self._sites


class _ShimModel:
    """Minimal model wrapper. TeNPy's lattice classes assume a repeating unit cell, which does
    not fit a chain whose first site (the cavity, dimension Nc^2) differs from the rest
    (emitters, dimension 4). The evolution engine only needs H_MPO and the site list, so we
    supply exactly that rather than contorting the model into a Lattice.
    """

    def __init__(self, sites, H_MPO):
        self.H_MPO = H_MPO
        self.lat = _Lat(sites)

    def estimate_RAM_saving_factor(self):
        return 1.0


def run_mps(M, Nc, wc, w, g, kappa, gdown, gphi, tmax, nt, chi_max, svd_min=1e-10,
            dt=None, substeps=8, order=2, approximation="II", compression="variational",
            verbose=False):
    """Evolve the vectorised density matrix and return the photon-number trajectory.

    Returns dict with times, n(t) = <a^dag a>, mean inversion z(t), Tr(rho)(t), the bond
    dimension actually reached, and wall-clock seconds.
    """
    sites, Lmpo = build_liouvillian(M, Nc, wc, w, g, kappa, gdown, gphi, scale=1j)
    psi = initial_state(sites, M, Nc)
    tr = trace_state(sites, M, Nc)

    ts = np.linspace(0.0, tmax, nt)
    # The measurement grid and the integration step are separate concerns. W_II carries an
    # O(t dt^order) error, so we substep between measurements rather than coarsening the
    # integrator to whatever the output grid happens to be.
    meas_dt = ts[1] - ts[0]
    if dt is None:
        dt = meas_dt / substeps
    n_sub = max(1, int(round(meas_dt / dt)))

    # W_II builds an MPO for exp(L dt); the generator is non-Hermitian, which is fine here.
    from tenpy.algorithms.mpo_evolution import ExpMPOEvolution
    options = {
        "dt": dt,
        "N_steps": n_sub,
        "order": order,
        "approximation": approximation,
        "start_time": 0.0,
        "trunc_params": {"chi_max": chi_max, "svd_min": svd_min},
        "preserve_norm": False,           # Tr(rho) is a diagnostic here, not something to enforce
        "compression_method": compression,
        "max_sweeps": 4,
    }

    eng = ExpMPOEvolution(psi, _ShimModel(sites, Lmpo), options)

    # measurement bras, built once
    cav, emi = cavity_ops(Nc), emitter_ops()
    bra_n = measurement_bra(sites, M, Nc, 0, cav["n"])
    bra_z = [measurement_bra(sites, M, Nc, i + 1, emi["sz"]) for i in range(M)]

    n_t, z_t, tr_t, chi_t = [], [], [], []
    t0 = time.time()
    for k, t in enumerate(ts):
        if k > 0:
            eng.run()
        cur = eng.psi
        n_t.append(bra_n.overlap(cur).real)
        z = sum(b.overlap(cur).real for b in bra_z) / M
        z_t.append(z)
        tr_t.append(trace_of(cur, tr).real)
        chi_t.append(max(cur.chi) if cur.chi else 1)
        if verbose and k % 20 == 0:
            print(f"    t={t:6.2f}  n={n_t[-1]:8.4f}  z={z:7.4f}  Tr={tr_t[-1]:.6f}  chi={chi_t[-1]}")
    secs = time.time() - t0

    return {
        "t": ts,
        "n": np.array(n_t),
        "z": np.array(z_t),
        "trace": np.array(tr_t),
        "chi": np.array(chi_t),
        "chi_max_reached": int(max(chi_t)),
        "seconds": secs,
    }
