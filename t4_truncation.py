"""
T4: conventional FOURTH-order cumulant truncation (kappa5 = 0) -- the fairness baseline (PROTOCOL "FOURTH-ORDER
CUMULANT BASELINE"). Does going one EVEN order up the standard hierarchy fix the disordered open-cavity burst
peak that 2nd-order overshoots and kappa4=0 T3 undershoots/diverges? Built exactly like the validated order-2
and order-3 engines, one order up; correctness is GATED, not assumed.

kappa5=0 closure of an order-5 moment via the general set-partition moment-cumulant relation:
    m_n = sum_{partitions pi of the n factors, |pi| >= 2}  (-1)^|pi| (|pi|-1)!  prod_{B in pi} m(B).
This is verified in __main__ to reproduce T3's hardcoded kappa4 terms exactly at n=4.

UNFORGEABLE GATE (< 1e-8 before any rollout is trusted): feeding ALL-exact moments, the assembled d<k>/dt must
equal the exact <L-dagger[k]>. Capped at M<=4 by pre-registration (the O(M^5) tracked set is part of the answer).
"""
import itertools
import math
import time
import numpy as np
import qutip as qt

from cumulant import enum_monos, mono_matrix, mono_order, factors_of
from t3_truncation import merge_key, kappa4_terms
from exact_tc import build_ops, build_H, collapse_ops, draw_disorder

WC = 1.0


def set_partitions(items):
    """All set partitions of a list, each as a list of blocks (lists)."""
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for p in set_partitions(rest):
        for i in range(len(p)):
            yield p[:i] + [[first] + p[i]] + p[i + 1:]
        yield [[first]] + p


def cumulant_closure_terms(key):
    """[(coeff, [block-monomial keys]) ...] closing the order-n moment with kappa_n = 0, for ANY order n>=2.
    m_n = sum_{|pi|>=2} (-1)^|pi| (|pi|-1)! prod m(block).  Reproduces kappa4_terms exactly at n=4."""
    facs = factors_of(key)
    n = len(facs)
    terms = []
    for part in set_partitions(list(range(n))):
        if len(part) < 2:
            continue
        coeff = ((-1) ** len(part)) * math.factorial(len(part) - 1)
        block_keys = [merge_key([facs[i] for i in block]) for block in part]
        terms.append((float(coeff), block_keys))
    return terms


class T4Engine:
    def __init__(self, M, Nc, wc, w, g, kappa=0.0, gdown=0.0, gphi=0.0, Fsafe=6):
        O = build_ops(M, Nc); O["Nc"] = Nc
        self.O, self.M, self.Nc = O, M, Nc
        self.H = build_H(O, wc, w, g)
        self.cops = collapse_ops(O, kappa=kappa, gdown=gdown, gphi=gphi)
        self.track = enum_monos(M, 4, pqmax=4)
        self.full = enum_monos(M, 5, pqmax=5)
        self.idx = {k: i for i, k in enumerate(self.full)}
        self.tidx = np.array([self.idx[k] for k in self.track])
        self.order5 = [k for k in self.full if mono_order(k) == 5]
        self.clos5 = {k: cumulant_closure_terms(k) for k in self.order5}
        fock = np.arange(Nc * (2 ** M)) // (2 ** M)
        FB, FK = np.meshgrid(fock, fock, indexing="ij")
        mask = ((FB <= Fsafe) & (FK <= Fsafe)).ravel()
        Bm = np.array([mono_matrix(k, O).full().ravel()[mask] for k in self.full]).T
        self.C = np.zeros((len(self.track), len(self.full)), complex)
        for a, k in enumerate(self.track):
            Ld = self._Ldag(mono_matrix(k, O)).full().ravel()[mask]
            self.C[a, :] = np.linalg.lstsq(Bm, Ld, rcond=None)[0]

    def _Ldag(self, X):
        out = 1j * (self.H * X - X * self.H)
        for c in self.cops:
            out = out + c.dag() * X * c - 0.5 * (c.dag() * c * X + X * c.dag() * c)
        return out

    def moments_from_rho(self, rho):
        return np.array([qt.expect(mono_matrix(k, self.O), rho) for k in self.full], complex)

    def full_from_tracked(self, mtrack):
        mt = {k: mtrack[i] for i, k in enumerate(self.track)}
        m = np.zeros(len(self.full), complex)
        for i, k in enumerate(self.full):
            if mono_order(k) <= 4:
                m[i] = mt[k]
            else:
                val = 0j
                for coeff, keys in self.clos5[k]:
                    prod = coeff
                    for kk in keys:
                        prod *= mt[kk]
                    val += prod
                m[i] = val
        return m

    def deriv(self, mtrack):
        return self.C @ self.full_from_tracked(mtrack)

    def gate_exact(self, rho):
        m_full = self.moments_from_rho(rho)
        d_assembled = self.C @ m_full
        d_exact = np.array([qt.expect(self._Ldag(mono_matrix(k, self.O)), rho) for k in self.track], complex)
        return float(np.max(np.abs(d_assembled - d_exact)))


if __name__ == "__main__":
    # (0) closure-formula self-check: cumulant_closure_terms == kappa4_terms at n=4 (order-4 key)
    M0 = 4
    k4 = (1, 1, (1, 2, 0, 0))                       # a^dag a sigma^+_0 sigma^-_1 : an order-4 monomial
    def norm(terms):
        return sorted((round(c, 9), tuple(sorted(map(tuple, ks)))) for c, ks in terms)
    ok_formula = norm(cumulant_closure_terms(k4)) == norm([(float(c), ks) for c, ks in kappa4_terms(k4)])
    print(f"closure formula reproduces T3 kappa4 terms at n=4: {ok_formula}")
    assert ok_formula, "cumulant_closure_terms disagrees with kappa4_terms -- formula bug"

    print("\nT4 (kappa5=0) engine GATE, M=3 (kappa=0.03, gphi=0.02, sigma=0.15, seed 5):")
    M, Nc = 3, 10
    w = draw_disorder(M, WC, 0.15, 5, "gauss")
    g = np.full(M, 0.7 / np.sqrt(M))
    t0 = time.time()
    eng = T4Engine(M, Nc, WC, w, g, kappa=0.03, gdown=0.02, gphi=0.02, Fsafe=6)
    print(f"  built: tracked={len(eng.track)} full={len(eng.full)} order5={len(eng.order5)} "
          f"({time.time()-t0:.0f}s)")
    psi0 = qt.tensor(qt.basis(Nc, 0), *[qt.basis(2, 0) for _ in range(M)])
    res = qt.mesolve(eng.H, psi0, np.linspace(0, 6, 4), c_ops=eng.cops, options={"progress_bar": False})
    ok = True
    for t, rho in zip([0, 2, 4, 6], res.states):
        err = eng.gate_exact(rho)
        p = err < 1e-8
        ok &= p
        print(f"  t={t}: max|d_assembled - d_exact| = {err:.2e}  {'PASS' if p else 'FAIL'}")
    print(f"T4 ENGINE GATE: {'PASS' if ok else 'FAIL'}")
