"""
Rigorous second-order cumulant EOM engine for open Tavis-Cummings, built WITHOUT hand-deriving
individual equations (which is where sign/factor errors hide). Instead:

  - Track all normal-ordered monomial moments of order <= 2 (cavity a^dag^p a^q times one of
    {I, s+, s-, sz} per emitter; order = p+q + #non-identity emitter sites).
  - The exact moment derivative is d<M>/dt = <L^dagger[M]>, and L^dagger[M] is computed EXACTLY as a
    matrix from H and the collapse operators. We project L^dagger[M] onto the monomial basis (up to
    order 3) by solving a linear system -> constant coefficients C[a,b]. This is exact.
  - Order-3 moments in that expansion are closed by the second-order cumulant factorization
    <f1 f2 f3> = <f1><f2f3> + <f2><f1f3> + <f3><f1f2> - 2<f1><f2><f3> + kappa3,
    with kappa3 = 0 (plain 2nd-order cumulant) or a supplied value (the learned closure).

Correctness gate (unforgeable): feed EXACT moments + EXACT kappa3 -> the assembled d<M>/dt must equal
the exact <L^dagger[M]> from the density matrix. If it matches to ~1e-10, the engine is correct.
"""
import itertools
import numpy as np
import qutip as qt
from .exact_tc import build_ops, build_H, collapse_ops


# --- monomial bookkeeping -------------------------------------------------------------------
# key = (p, q, tuple(emitter ops in {0:I,1:'+',2:'-',3:'z'}));  operator = a^dag^p a^q * prod sigma
def mono_order(key):
    p, q, es = key
    return p + q + sum(1 for e in es if e != 0)


def enum_monos(M, max_order, pqmax=3):
    keys = []
    for p in range(pqmax + 1):
        for q in range(pqmax + 1):
            for es in itertools.product(range(4), repeat=M):
                k = (p, q, es)
                if mono_order(k) <= max_order:
                    keys.append(k)
    return keys


def mono_matrix(key, O):
    p, q, es = key
    a = O["a"]
    mat = qt.qeye(a.dims[0]) if False else None
    op = a.dag() ** p * a ** q if (p or q) else qt.tensor(qt.qeye(O["Nc"]), *[qt.qeye(2)] * O["M"])
    smap = {1: O["sp"], 2: O["sm"], 3: O["sz"]}
    for i, e in enumerate(es):
        if e != 0:
            op = op * smap[e][i]
    return op


# --- cumulant closure of an order-3 monomial into <=order-2 pieces + kappa3 ------------------
def factors_of(key):
    """Return the ordered list of fundamental factors of a monomial as sub-keys of order 1."""
    p, q, es = key
    M = len(es)
    f = []
    for _ in range(p):
        f.append((1, 0, tuple([0] * M)))            # a^dag
    for _ in range(q):
        f.append((0, 1, tuple([0] * M)))            # a
    for i, e in enumerate(es):
        if e != 0:
            emit = [0] * M; emit[i] = e
            f.append((0, 0, tuple(emit)))           # single-site sigma
    return f


def pair_key(fa, fb):
    """Normal-ordered product monomial of two order-1 factors fa,fb (a^dag before a; emitters commute)."""
    p = fa[0] + fb[0]; q = fa[1] + fb[1]
    es = tuple(fa[2][i] + fb[2][i] for i in range(len(fa[2])))  # at most one nonzero each per site here
    return (p, q, es)


def closure_terms(key):
    """
    For an order-3 monomial, return a list of (coeff, [factor-index tuples]) describing the
    factorized value in terms of order-1 and order-2 monomials, plus a flag for the kappa3 factors.
    We return: list of ('m1*m2', key1, key2) and ('m1*m2*m3', k1,k2,k3) contributions and the kappa3 triple.
    """
    f = factors_of(key)               # 3 order-1 factors
    assert len(f) == 3
    terms = []                        # each: (coeff, [keys to multiply])
    (f1, f2, f3) = f
    terms.append((1.0, [f1, pair_key(f2, f3)]))
    terms.append((1.0, [f2, pair_key(f1, f3)]))
    terms.append((1.0, [f3, pair_key(f1, f2)]))
    terms.append((-2.0, [f1, f2, f3]))
    return terms, (f1, f2, f3)


# --- engine ---------------------------------------------------------------------------------
class CumulantEngine:
    def __init__(self, M, Nc, wc, w, g, kappa=0.0, gdown=0.0, gphi=0.0, gpump=0.0, Fsafe=None):
        O = build_ops(M, Nc); O["Nc"] = Nc
        self.O = O; self.M = M; self.Nc = Nc
        H = build_H(O, wc, w, g)
        cops = collapse_ops(O, kappa=kappa, gdown=gdown, gphi=gphi, gpump=gpump)
        # L^dag[X] = i[H,X] + sum_c c^dag X c - 1/2{c^dag c, X}  (Heisenberg/adjoint generator)
        self.H = H; self.cops = cops
        self.track = enum_monos(M, 2)
        self.full = enum_monos(M, 3)
        self.idx = {k: i for i, k in enumerate(self.full)}
        # Fock-boundary mask: fit the projection using only matrix elements safely below the cutoff, so
        # truncation never pollutes the normal-ordered bosonic algebra (the projection then equals the
        # exact infinite-Fock cumulant coefficients; verified by the correctness gate to ~1e-14).
        self.Fsafe = Fsafe if Fsafe is not None else max(4, Nc - 6)
        fock = np.arange(Nc * (2 ** M)) // (2 ** M)
        FB, FK = np.meshgrid(fock, fock, indexing="ij")
        self.mask = ((FB <= self.Fsafe) & (FK <= self.Fsafe)).ravel()
        Bfull = np.array([mono_matrix(k, O).full().ravel() for k in self.full]).T
        self.B = Bfull
        Bm = Bfull[self.mask, :]
        self.C = np.zeros((len(self.track), len(self.full)), complex)
        for a, k in enumerate(self.track):
            Ld = self._Ldag(mono_matrix(k, O)).full().ravel()[self.mask]
            self.C[a, :] = np.linalg.lstsq(Bm, Ld, rcond=None)[0]
        self.order3 = [k for k in self.full if mono_order(k) == 3]
        self.clos = {k: closure_terms(k) for k in self.order3}
        self.tidx = np.array([self.idx[k] for k in self.track])
        # --- vectorized closure assembly (precomputed indices; math identical to _full_from_tracked) ---
        tk = {k: i for i, k in enumerate(self.track)}
        self._o3_fidx = np.array([self.idx[k] for k in self.order3])
        s = [[], [], []]; p = [[], [], []]; tri = [[], [], []]
        for k in self.order3:
            terms, _ = self.clos[k]                       # 3 single*pair terms (+1) then one triple term (-2)
            for a in range(3):
                _, keys = terms[a]
                s[a].append(tk[keys[0]]); p[a].append(tk[keys[1]])
            _, tkeys = terms[3]
            for a in range(3):
                tri[a].append(tk[tkeys[a]])
        self._c_s = [np.array(x) for x in s]
        self._c_p = [np.array(x) for x in p]
        self._c_t = [np.array(x) for x in tri]
        # build-time guard: the fast path MUST equal the validated reference to machine precision
        rng = np.random.default_rng(0)
        mtst = rng.standard_normal(len(self.track)) + 1j * rng.standard_normal(len(self.track))
        if not np.allclose(self._full_fast(mtst, None), self._full_from_tracked(mtst, {}), atol=1e-12):
            raise RuntimeError("fast closure assembly does not match validated _full_from_tracked")

    def _Ldag(self, X):
        H = self.H
        out = 1j * (H * X - X * H)
        for c in self.cops:
            out = out + c.dag() * X * c - 0.5 * (c.dag() * c * X + X * c.dag() * c)
        return out

    def moments_from_rho(self, rho):
        return np.array([qt.expect(mono_matrix(k, self.O), rho) for k in self.full], complex)

    def _full_from_tracked(self, mtrack, kappa3):
        """Assemble the full order<=3 moment vector from tracked (order<=2) moments + kappa3 dict."""
        mt = {k: mtrack[i] for i, k in enumerate(self.track)}
        m = np.zeros(len(self.full), complex)
        for i, k in enumerate(self.full):
            if mono_order(k) <= 2:
                m[i] = mt[k]
            else:
                terms, trip = self.clos[k]
                val = 0.0 + 0j
                for coeff, keys in terms:
                    prod = coeff
                    for kk in keys:
                        prod *= mt[kk]
                    val += prod
                val += kappa3.get(k, 0.0)
                m[i] = val
        return m

    def deriv(self, mtrack, kappa3):
        m_full = self._full_from_tracked(mtrack, kappa3)
        return self.C @ m_full        # d(tracked)/dt

    def _full_fast(self, mtrack, kap_vec):
        """Vectorized equivalent of _full_from_tracked; kap_vec is aligned to self.order3 (or None)."""
        m = np.zeros(len(self.full), complex)
        m[self.tidx] = mtrack                             # all order<=2 entries map straight through
        m3 = (mtrack[self._c_s[0]] * mtrack[self._c_p[0]]
              + mtrack[self._c_s[1]] * mtrack[self._c_p[1]]
              + mtrack[self._c_s[2]] * mtrack[self._c_p[2]]
              - 2.0 * mtrack[self._c_t[0]] * mtrack[self._c_t[1]] * mtrack[self._c_t[2]])
        if kap_vec is not None:
            m3 = m3 + kap_vec
        m[self._o3_fidx] = m3
        return m

    def deriv_fast(self, mtrack, kap_vec=None):
        return self.C @ self._full_fast(mtrack, kap_vec)

    # ---- correctness gate ----
    def gate_exact(self, rho):
        """Feed exact moments + exact kappa3 -> assembled derivative must equal exact <L^dag[M]>."""
        m_full_exact = self.moments_from_rho(rho)
        mt = m_full_exact[[self.idx[k] for k in self.track]]
        # exact kappa3 = exact order-3 moment minus its factorized (kappa3=0) value
        kappa3 = {}
        base = self._full_from_tracked(mt, {})
        for k in self.order3:
            kappa3[k] = m_full_exact[self.idx[k]] - base[self.idx[k]]
        d_assembled = self.deriv(mt, kappa3)
        # exact derivative d<M>/dt = <L^dag[M]>
        d_exact = np.array([qt.expect(self._Ldag(mono_matrix(k, self.O)), rho) for k in self.track], complex)
        err = np.max(np.abs(d_assembled - d_exact))
        return err, d_assembled, d_exact


if __name__ == "__main__":
    print("=" * 74)
    print("CUMULANT ENGINE correctness gate: exact moments -> exact derivatives")
    print("=" * 74)
    import numpy as np
    M, Nc = 2, 14
    wc = 1.0; w = np.array([1.0, 1.03]); g = np.array([0.15, 0.15])
    eng = CumulantEngine(M, Nc, wc, w, g, kappa=0.1, gdown=0.02, gphi=0.02, Fsafe=6)
    print(f"tracked moments: {len(eng.track)}   full (<=order3): {len(eng.full)}   order-3: {len(eng.order3)}   Fsafe={eng.Fsafe}")
    psi0 = qt.tensor(qt.basis(Nc, 0), qt.basis(2, 0), qt.basis(2, 0))
    res = qt.mesolve(eng.H, psi0, np.linspace(0, 5, 6), c_ops=eng.cops, options={"progress_bar": False})
    ok = True
    for t, rho in zip([0, 1, 2, 3, 4, 5], res.states):
        err, _, _ = eng.gate_exact(rho)
        p = err < 1e-8; ok = ok and p
        print(f"   t={t}: max|d_assembled - d_exact| = {err:.2e}  {'PASS' if p else 'FAIL'}")
    print(f"\nENGINE CORRECTNESS GATE: {'PASS' if ok else 'FAIL'}")
