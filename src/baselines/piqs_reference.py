"""
Permutation-invariant EXACT reference for the open Tavis-Cummings model at ZERO disorder.

WHY THIS EXISTS
---------------
Every exact reference in this repository is capped by 2^M: mesolve reaches M <= 5 and seeded
trajectories reach M <= 10. The learned closure's measured instability starts at M >= 12, which is
precisely where no reference exists, so the defect could never be characterised -- only detected.

When the emitters are IDENTICAL the Liouvillian is permutation invariant and the dynamics closes on
the Dicke basis, whose dimension grows POLYNOMIALLY rather than exponentially:

    M      Dicke states      2^M
    12         49            4,100
    20        121            1.05e6
    50        676            1.13e15
    100     2,601            1.27e30

That buys exact references at sizes the rest of the repository cannot touch. It is implemented in
QuTiP as qutip.piqs (Shammah et al., Phys. Rev. A 98, 063815 (2018)).

WHAT IT DOES NOT BUY, AND THIS IS THE WHOLE POINT
-------------------------------------------------
Disorder BREAKS permutation invariance, so this method applies at sigma = 0 only. It therefore
cannot validate the disordered cells the learned closure was built for. What it can do is separate
two hypotheses that are currently confounded at M >= 12:

    (a) the instability is a SIZE pathology  -> it should appear at sigma = 0 as well
    (b) the instability needs DISORDER       -> it should be absent at sigma = 0

That is a real fork, it is cheap, and it is answerable. It also sharpens the niche this work claims:
at sigma = 0 permutation invariance solves any M; at sigma > 0 exact methods stop at M <= 10; the
region sigma > 0 AND M >= 12 is the one with no working method.

MODEL MAPPING (identical emitters, w_i = wc, g_i = g_c/sqrt(M))
---------------------------------------------------------------
    sum_i |e><e|_i          ->  jz + M/2
    sum_i sigma^+_i         ->  jp
    sum_i sigma^-_i         ->  jm
    H = wc a'a + w (jz + M/2) + g (a jp + a' jm)

Cavity loss stays an ordinary QuTiP dissipator. Local emission and local dephasing are supplied by
the Dicke class. Its rate conventions are NOT assumed: calibrate_piqs_rates() measures them against
closed-form single-emitter decay, exactly as check_conventions.py does for the main model.
"""
import numpy as np
import qutip as qt
from qutip.piqs.piqs import Dicke, jspin, num_dicke_states, excited


def calibrate_piqs_rates(verbose=True):
    """Measure how PIQS's `emission` and `dephasing` map onto this project's conventions.

    This project uses  gdown D[sigma^-_i]  and  (gphi/2) D[sigma^z_i], which give
        <sp sm> decaying at gdown,  |<sm>| decaying at gdown/2 + gphi.
    Returns (emission_factor, dephasing_factor) such that
        Dicke(emission=gdown*f_e, dephasing=gphi*f_d)
    reproduces those rates. A factor of 1.0 means the conventions already agree.
    """
    N = 1
    T = np.linspace(0, 6, 13)
    jz, jp, jm = jspin(N, "z"), jspin(N, "+"), jspin(N, "-")

    def decay_rates(emission, dephasing):
        L = Dicke(N, emission=emission, dephasing=dephasing).liouvillian()
        # start in an equal superposition so both population and coherence are nonzero
        psi = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
        rho0 = qt.Qobj(qt.ket2dm(psi).full(), dims=jz.dims)
        res = qt.mesolve(L, rho0, T, options={"progress_bar": False, "atol": 1e-13, "rtol": 1e-11})
        pop = np.array([qt.expect(jz, s).real + 0.5 for s in res.states])   # <|e><e|>
        coh = np.array([abs(qt.expect(jm, s)) for s in res.states])
        # fit log-linear slope
        r_pop = -np.polyfit(T, np.log(np.clip(pop, 1e-300, None)), 1)[0]
        r_coh = -np.polyfit(T, np.log(np.clip(coh, 1e-300, None)), 1)[0]
        return r_pop, r_coh

    r_pop, _ = decay_rates(1.0, 0.0)          # pure emission at unit coefficient
    _, r_coh = decay_rates(0.0, 1.0)          # pure dephasing at unit coefficient
    f_e = 1.0 / r_pop                          # want <sp sm> to decay at gdown
    f_d = 1.0 / r_coh                          # want |<sm>| to decay at gphi
    if verbose:
        print(f"   PIQS emission=1.0  -> population decay rate {r_pop:.6f}  => factor {f_e:.6f}")
        print(f"   PIQS dephasing=1.0 -> coherence  decay rate {r_coh:.6f}  => factor {f_d:.6f}")
    return f_e, f_d


def build_piqs_liouvillian(M, Nc, wc, w0, g, kappa, gdown, gphi, rate_factors=None):
    """Total Liouvillian for cavity + M identical emitters, in (Fock x Dicke) space."""
    f_e, f_d = rate_factors if rate_factors is not None else calibrate_piqs_rates(verbose=False)
    nds = num_dicke_states(M)
    jz, jp, jm = jspin(M, "z"), jspin(M, "+"), jspin(M, "-")
    a = qt.destroy(Nc)
    id_c = qt.qeye(Nc)
    id_s = qt.qeye(nds)
    # lift to the product space, matching the Dicke-basis dims PIQS uses
    A = qt.tensor(a, id_s)
    JZ, JP, JM = qt.tensor(id_c, jz), qt.tensor(id_c, jp), qt.tensor(id_c, jm)
    H = wc * A.dag() * A + w0 * (JZ + 0.5 * M * qt.tensor(id_c, id_s)) + g * (A * JP + A.dag() * JM)
    # cavity loss as an ordinary dissipator; emitter dissipation from the Dicke superoperator
    L = qt.liouvillian(H, [np.sqrt(kappa) * A]) if kappa > 0 else qt.liouvillian(H)
    D_spin = Dicke(M, emission=gdown * f_e, dephasing=gphi * f_d).liouvillian()
    L = L + qt.super_tensor(qt.to_super(id_c), D_spin)
    return L, A


def burst_piqs(M, Nc, wc, gc, kappa, gdown, gphi, tmax=40.0, nt=200, rate_factors=None,
               atol=1e-12, rtol=1e-10):
    """Exact zero-disorder burst trajectory <a'a>(t). Returns (t, n, peak_flux, t_peak)."""
    g = gc / np.sqrt(M)
    L, A = build_piqs_liouvillian(M, Nc, wc, wc, g, kappa, gdown, gphi, rate_factors)
    nds = num_dicke_states(M)
    rho0 = qt.tensor(qt.ket2dm(qt.basis(Nc, 0)), excited(M))
    rho0 = qt.Qobj(rho0.full(), dims=[[Nc, nds], [Nc, nds]])
    t = np.linspace(0, tmax, nt)
    res = qt.mesolve(L, rho0, t, e_ops=[A.dag() * A],
                     options={"progress_bar": False, "atol": atol, "rtol": rtol})
    n = np.asarray(res.expect[0]).real
    k = int(np.argmax(n))
    return t, n, kappa * n[k], t[k]
