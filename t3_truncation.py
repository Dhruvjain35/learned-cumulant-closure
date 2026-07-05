"""
T3: conventional THIRD-order cumulant truncation (kappa4 = 0) -- the "why not just track third order?"
baseline (PROTOCOL Phase B; capped effort, M <= 4 by pre-registration; the O(M^4) growth of the tracked
set is itself part of the answer).

Same construction discipline as the validated order-2 engine: track all normal-ordered monomials of
order <= 3, project L-dagger[k] onto the order <= 4 basis with the Fock-boundary-masked least squares,
close order-4 monomials by the standard kappa4=0 factorization over the ordered fundamental factors:
    m(abcd) = sum_4 m1*m3 + sum_3 m2*m2 - 2*sum_6 m2*m1*m1 + 6*m1^4.
UNFORGEABLE GATE (required < 1e-8 before any rollout is trusted): feeding ALL-exact moments, the
assembled d<k>/dt must equal the exact <L-dagger[k]>.
"""
import itertools
import time
import numpy as np
import qutip as qt

from cumulant import enum_monos, mono_matrix, mono_order, factors_of
from exact_tc import build_ops, build_H, collapse_ops, draw_disorder

WC = 1.0


def merge_key(facs):
    """Normal-ordered monomial key for a set of order-1 factor keys (generalizes pair_key)."""
    p = sum(f[0] for f in facs)
    q = sum(f[1] for f in facs)
    M = len(facs[0][2])
    es = [0] * M
    for f in facs:
        for i, e in enumerate(f[2]):
            if e:
                assert es[i] == 0, "site collision in merge_key"
                es[i] = e
    return (p, q, tuple(es))


def kappa4_terms(key):
    """[(coeff, [subset-monomial keys to multiply])] for the kappa4=0 closure of an order-4 monomial."""
    f = factors_of(key)
    assert len(f) == 4
    a, b, c, d = f
    terms = []
    for x, rest in [(a, (b, c, d)), (b, (a, c, d)), (c, (a, b, d)), (d, (a, b, c))]:
        terms.append((1.0, [merge_key([x]), merge_key(list(rest))]))                 # 4 x m1*m3
    for (x, y), (u, v) in [((a, b), (c, d)), ((a, c), (b, d)), ((a, d), (b, c))]:
        terms.append((1.0, [merge_key([x, y]), merge_key([u, v])]))                  # 3 x m2*m2
    for (x, y), (u, v) in [((a, b), (c, d)), ((a, c), (b, d)), ((a, d), (b, c)),
                           ((b, c), (a, d)), ((b, d), (a, c)), ((c, d), (a, b))]:
        terms.append((-2.0, [merge_key([x, y]), merge_key([u]), merge_key([v])]))    # -2 x 6 x m2*m1*m1
    terms.append((6.0, [merge_key([a]), merge_key([b]), merge_key([c]), merge_key([d])]))
    return terms


class T3Engine:
    def __init__(self, M, Nc, wc, w, g, kappa=0.0, gdown=0.0, gphi=0.0, Fsafe=6):
        O = build_ops(M, Nc); O["Nc"] = Nc
        self.O, self.M, self.Nc = O, M, Nc
        self.H = build_H(O, wc, w, g)
        self.cops = collapse_ops(O, kappa=kappa, gdown=gdown, gphi=gphi)
        self.track = enum_monos(M, 3, pqmax=3)
        self.full = enum_monos(M, 4, pqmax=4)
        self.idx = {k: i for i, k in enumerate(self.full)}
        self.tidx = np.array([self.idx[k] for k in self.track])
        self.order4 = [k for k in self.full if mono_order(k) == 4]
        self.clos4 = {k: kappa4_terms(k) for k in self.order4}
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
            if mono_order(k) <= 3:
                m[i] = mt[k]
            else:
                val = 0j
                for coeff, keys in self.clos4[k]:
                    prod = coeff
                    for kk in keys:
                        prod *= mt[kk]
                    val += prod
                m[i] = val
        return m

    def deriv(self, mtrack):
        return self.C @ self.full_from_tracked(mtrack)

    def gate_exact(self, rho):
        m_full = self.moments_from_rho(rho)                    # ALL exact incl. order 4
        d_assembled = self.C @ m_full
        d_exact = np.array([qt.expect(self._Ldag(mono_matrix(k, self.O)), rho)
                            for k in self.track], complex)
        return float(np.max(np.abs(d_assembled - d_exact)))


if __name__ == "__main__":
    print("T3 (kappa4=0) engine gate, M=3 (kappa=0.15, gphi=0.02, sigma=0.15, train-rng seed 5):")
    M, Nc = 3, 10
    w = draw_disorder(M, WC, 0.15, 5, "gauss")
    g = np.full(M, 0.7 / np.sqrt(M))
    t0 = time.time()
    eng = T3Engine(M, Nc, WC, w, g, kappa=0.15, gdown=0.02, gphi=0.02, Fsafe=6)
    print(f"  built: tracked={len(eng.track)} full={len(eng.full)} order4={len(eng.order4)} "
          f"({time.time()-t0:.0f}s)")
    psi0 = qt.tensor(qt.basis(Nc, 0), *[qt.basis(2, 0) for _ in range(M)])
    res = qt.mesolve(eng.H, psi0, np.linspace(0, 6, 4), c_ops=eng.cops, options={"progress_bar": False})
    ok = True
    for t, rho in zip([0, 2, 4, 6], res.states):
        err = eng.gate_exact(rho)
        p = err < 1e-8
        ok &= p
        print(f"  t={t}: max|d_assembled - d_exact| = {err:.2e}  {'PASS' if p else 'FAIL'}")
    print(f"T3 ENGINE GATE: {'PASS' if ok else 'FAIL'}")
