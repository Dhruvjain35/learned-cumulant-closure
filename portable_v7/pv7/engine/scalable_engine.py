"""
Scalable second-order cumulant EOM for open Tavis-Cummings, valid at ARBITRARY molecule number M.

Why this exists: the dense CumulantEngine builds the exact EOM by projecting operators on the full
Nc*2^M Hilbert space, which OOMs at M >= 6. But step1_locality.py proved (on the already-1e-14-validated
dense engine) that the exact EOM is:
  (A) SITE-LOCAL: every term in d<k>/dt lives on support(k) plus at most one extra emitter site, and
  (B) PERMUTATION-STRUCTURED: with the model's uniform coupling/loss it is invariant under relabeling sites.
So the whole EOM can be assembled from tiny subsystem computations (cavity + <=3 qubits) placed onto M sites.

Decomposition used (exact; L^dagger is linear in the physical parameters):
  L^dagger = L_freq(wc, w_i)  +  L_struct(g, kappa, gdown, gphi, gpump)
  - L_freq is purely DIAGONAL: d<k>/dt gets  i*( wc*(p-q) + sum_i w_i*s_i(k) ) * <k>,  where s_i = +1 if
    site i carries sigma^+ in k, -1 if sigma^-, 0 otherwise. Written analytically for any M. No projection.
  - L_struct uses only the UNIFORM parameters (g the same on every site, and the uniform loss rates), so it
    is permutation-symmetric. Its structure is computed ONCE on reference systems of s+1 qubits (s = number
    of support sites of the monomial, s in {0,1,2}) with wc=w=0, then placed:
      * terms whose extra qubit is identity  -> on-support, placed once onto support(k);
      * terms whose extra qubit carries an operator -> summed over every emitter j not in support(k)
        (same coefficient for all j, since g is uniform and, as proved, the extra site enters only via g).

Correctness is NOT taken on faith: build_C() assembles the full coefficient matrix and is asserted equal
to the validated dense C at M = 2,3,4,5 (validate_scalable.py). Only after that passes is it trusted at
large M, where no dense engine can be built to check against. No hallucination, nothing hand-derived.
"""
import itertools
import numpy as np
from .cumulant import CumulantEngine, enum_monos, mono_order

# emitter code: 0 = identity, 1 = sigma^+, 2 = sigma^-, 3 = sigma^z
SPLUS, SMINUS, SZ = 1, 2, 3
FULL_CAP = 7          # above this M, do not materialize the full order-3 basis (4^M enumeration is infeasible)


def enum_sparse(M, K, pqmax=3):
    """All normal-ordered monomials of order <= K, generated sparsely (only <=K non-identity factors),
    so it is O(M^K) instead of enum_monos' 4^M. Same SET as enum_monos(M,K); order may differ."""
    keys = []
    for p in range(pqmax + 1):
        for q in range(pqmax + 1):
            if p + q > K:
                continue
            budget = K - (p + q)
            for nsites in range(budget + 1):
                for sites in itertools.combinations(range(M), nsites):
                    for ops in itertools.product((SPLUS, SMINUS, SZ), repeat=nsites):
                        es = [0] * M
                        for st, op in zip(sites, ops):
                            es[st] = op
                        keys.append((p, q, tuple(es)))
    return keys


def support(key):
    _, _, es = key
    return tuple(i for i, e in enumerate(es) if e != 0)


def freq_sign(e):
    return 1.0 if e == SPLUS else (-1.0 if e == SMINUS else 0.0)


class ScalableEngine:
    def __init__(self, M, Nc, wc, w, g, kappa=0.0, gdown=0.0, gphi=0.0, gpump=0.0, Fsafe=6):
        self.M, self.Nc, self.wc = M, Nc, float(wc)
        self.w = np.asarray(w, float)
        g = np.asarray(g, float)
        assert np.allclose(g, g[0]), "ScalableEngine assumes uniform coupling g (only w may be disordered)"
        self.g = float(g[0])
        self.kappa, self.gdown, self.gphi, self.gpump = kappa, gdown, gphi, gpump
        self.Fsafe = Fsafe

        # small M: use enum_monos ordering so build_C aligns index-for-index with the dense engine (validation).
        # large M: enum_monos is 4^M and infeasible, so enumerate the tracked (order<=2) basis sparsely and
        # never materialize the full order-3 basis (the streaming EOM only needs order-3 values via the closure).
        if M <= FULL_CAP:
            self.track = enum_monos(M, 2)
            self.full = enum_monos(M, 3)
            self.idx = {k: i for i, k in enumerate(self.full)}
            self.order3 = [k for k in self.full if mono_order(k) == 3]
        else:
            self.track = enum_sparse(M, 2)
            self.full = None
            self.idx = None
            self.order3 = None
        self.tpos = {k: i for i, k in enumerate(self.track)}

        # reference engines for s = 0,1,2 support sites: built on (s+1) qubits, wc=0, w=0, uniform g/loss.
        self.smax = min(2, M)
        self.ref = {}
        self.ref_tpos = {}
        for s in range(0, self.smax + 1):
            nq = s + 1
            eng = CumulantEngine(nq, Nc, 0.0, np.zeros(nq), np.full(nq, self.g),
                                 kappa=kappa, gdown=gdown, gphi=gphi, gpump=gpump, Fsafe=Fsafe)
            self.ref[s] = eng
            self.ref_tpos[s] = {kk: i for i, kk in enumerate(eng.track)}

    # ---- map a reference-system monomial back to a global monomial -------------------------------
    def _place(self, refkey, sites, extra_site):
        """refkey lives on the reference qubits (slots 0..s-1 = support 'sites', slot s = 'extra').
        Return the global key (p,q,es) with slot t -> sites[t], and the extra slot -> extra_site."""
        p, q, esl = refkey
        s = len(sites)
        es = [0] * self.M
        for t in range(s):
            if esl[t] != 0:
                es[sites[t]] = esl[t]
        if len(esl) > s and esl[s] != 0:                 # extra slot occupied
            if extra_site is None:
                return None                              # no site available to place it -> term absent
            es[extra_site] = esl[s]
        return (p, q, tuple(es))

    def build_C(self):
        """Assemble the full (len(track) x len(full)) coefficient matrix by placement. For validation/small M."""
        C = np.zeros((len(self.track), len(self.full)), complex)
        for a, k in enumerate(self.track):
            S = support(k)
            s = len(S)
            ref = self.ref[s]
            # embed k into the reference: support sites -> slots 0..s-1, extra slot -> identity
            p, q, es = k
            esl = [0] * (s + 1)
            for t, site in enumerate(S):
                esl[t] = es[site]
            refkey = (p, q, tuple(esl))
            ra = self.ref_tpos[s][refkey]
            row = ref.C[ra]
            others = [j for j in range(self.M) if j not in S]
            for b, rk in enumerate(ref.full):
                c = row[b]
                if c == 0:
                    continue
                _, _, resl = rk
                extra_used = (len(resl) > s and resl[s] != 0)
                if not extra_used:
                    gk = self._place(rk, S, None)
                    C[a, self.idx[gk]] += c
                else:
                    for j in others:                     # same coeff for every extra site (uniform g)
                        gk = self._place(rk, S, j)
                        C[a, self.idx[gk]] += c
            # analytic diagonal frequency term  i*( wc*(p-q) + sum_i w_i s_i )
            diag = 1j * (self.wc * (p - q) + sum(self.w[i] * freq_sign(es[i]) for i in S))
            C[a, self.idx[k]] += diag
        return C

    # ---- streaming EOM for large M (never materializes the full C) --------------------------------
    def precompute(self):
        """Cache per-tracked-monomial placement templates so deriv is evaluated without a full C.
        Each template: (k, support S, others=sites not in S, diagonal freq coeff,
                        on-support terms [(coeff, global_key)], extra-site terms [(coeff, refkey)])."""
        self.templates = []
        for k in self.track:
            S = support(k)
            s = len(S)
            ref = self.ref[s]
            p, q, es = k
            esl = [0] * (s + 1)
            for t, site in enumerate(S):
                esl[t] = es[site]
            row = ref.C[self.ref_tpos[s][(p, q, tuple(esl))]]
            onsup, extra = [], []
            for b, rk in enumerate(ref.full):
                c = row[b]
                if c == 0:
                    continue
                _, _, resl = rk
                if len(resl) > s and resl[s] != 0:
                    extra.append((c, rk))                       # extra qubit used -> place per site j not in S
                else:
                    onsup.append((c, self._place(rk, S, None)))  # on-support -> fixed global key
            others = tuple(j for j in range(self.M) if j not in S)
            diag = 1j * (self.wc * (p - q) + sum(self.w[i] * freq_sign(es[i]) for i in S))
            self.templates.append((k, S, others, diag, onsup, extra))
        return self

    def order3_keys_needed(self):
        """All distinct order-3 monomials the streaming deriv will request (for batching the closure)."""
        keys = set()
        for k, S, others, diag, onsup, extra in self.templates:
            for c, gk in onsup:
                if mono_order(gk) == 3:
                    keys.add(gk)
            for c, rk in extra:
                for j in others:
                    gk = self._place(rk, S, j)
                    if mono_order(gk) == 3:
                        keys.add(gk)
        return keys

    def deriv_stream(self, lookup):
        """d<track>/dt using a moment lookup(gkey)->value. lookup must handle order-3 (via closure).
        Reference implementation (slow, pure Python); deriv_compiled below is the fast equivalent."""
        out = np.zeros(len(self.track), complex)
        for a, (k, S, others, diag, onsup, extra) in enumerate(self.templates):
            val = diag * lookup(k)
            for c, gk in onsup:
                val += c * lookup(gk)
            for c, rk in extra:
                for j in others:
                    val += c * lookup(self._place(rk, S, j))
            out[a] = val
        return out

    # ---- compiled (vectorized) EOM: same math as deriv_stream, expressed as sparse matmuls ----------
    def compile(self):
        """Precompute sparse operators so deriv is d = T2 @ m + T3 @ (v3_factorized(m) + kappa3).
        T2 gathers all order<=2 contributions (diagonal freq + on-support + extra-site), T3 gathers the
        order-3 contributions, and v3_factorized applies the cumulant factorization to the tracked moments.
        o3_keys is the ordered list of order-3 monomials the closure must predict kappa3 for."""
        import scipy.sparse as sp
        from .cumulant import closure_terms
        if not hasattr(self, "templates"):
            self.precompute()
        nt = len(self.track)
        o3_col = {}
        t2r, t2c, t2d = [], [], []
        t3r, t3c, t3d = [], [], []

        def col3(gk):
            ci = o3_col.get(gk)
            if ci is None:
                ci = len(o3_col)
                o3_col[gk] = ci
            return ci

        def add(a, c, gk):
            if mono_order(gk) <= 2:
                t2r.append(a); t2c.append(self.tpos[gk]); t2d.append(c)
            else:
                t3r.append(a); t3c.append(col3(gk)); t3d.append(c)

        for a, (k, S, others, diag, onsup, extra) in enumerate(self.templates):
            t2r.append(a); t2c.append(self.tpos[k]); t2d.append(diag)
            for c, gk in onsup:
                add(a, c, gk)
            for c, rk in extra:
                for j in others:
                    add(a, c, self._place(rk, S, j))

        self.T2 = sp.csr_matrix((t2d, (t2r, t2c)), shape=(nt, nt), dtype=complex)
        K3 = len(o3_col)
        self.T3 = sp.csr_matrix((t3d, (t3r, t3c)), shape=(nt, K3), dtype=complex)
        self.o3_keys = [None] * K3
        for gk, ci in o3_col.items():
            self.o3_keys[ci] = gk
        cs, cp, ct = [[], [], []], [[], [], []], [[], [], []]
        for gk in self.o3_keys:
            terms, _ = closure_terms(gk)          # [ (1,[f,pair]) x3 , (-2,[f1,f2,f3]) ]
            for r in range(3):
                _, keys = terms[r]
                cs[r].append(self.tpos[keys[0]]); cp[r].append(self.tpos[keys[1]])
            _, tk = terms[3]
            for r in range(3):
                ct[r].append(self.tpos[tk[r]])
        self._cs = [np.array(x, int) for x in cs]
        self._cp = [np.array(x, int) for x in cp]
        self._ct = [np.array(x, int) for x in ct]
        return self

    def v3_factorized(self, m):
        """Cumulant factorization value (kappa3=0) for every key in o3_keys, from tracked moments m."""
        if len(self.o3_keys) == 0:
            return np.zeros(0, complex)
        return (m[self._cs[0]] * m[self._cp[0]] + m[self._cs[1]] * m[self._cp[1]]
                + m[self._cs[2]] * m[self._cp[2]]
                - 2.0 * m[self._ct[0]] * m[self._ct[1]] * m[self._ct[2]])

    def deriv_compiled(self, m, kap_vec=None):
        v3 = self.v3_factorized(m)
        if kap_vec is not None:
            v3 = v3 + kap_vec
        return self.T2.dot(m) + self.T3.dot(v3)
