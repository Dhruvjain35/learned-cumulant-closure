"""
STEP 1 (make-or-break for the scalable engine): is the validated dense cumulant EOM site-local?

The scalable (large-M) engine will assemble the second-order cumulant equations of motion from tiny
subsystem computations (cavity + at most 3 emitter sites) instead of the full Nc*2^M operator algebra.
That is only valid if the exact generator has two structural properties, which we verify here directly
on the ALREADY-VALIDATED dense engine (whose C matrix is correct to ~1e-14 by the gate):

  (A) SUPPORT LOCALITY: for every nonzero coefficient C[k, j] (the weight of full monomial j in
      d<k>/dt), the emitter support of j is contained in support(k) plus AT MOST ONE extra site.
      Reason it should hold: L^dagger applies the generator once; H_int is a sum of single-site
      couplings, so one commutator introduces at most one new emitter site; dissipators are single-site.

  (B) PERMUTATION STRUCTURE: with identical emitters, C is invariant under relabeling the emitter sites
      (applied to both the row monomial and the column monomials). This is what lets one local computation
      per canonical type serve every site assignment and every M.

No hallucination: every claim is a numerical assertion on the validated dense C. If either fails, the
scalable design is not valid as-is and must be revised. Nothing is assumed.
"""
import numpy as np
from cumulant import CumulantEngine, mono_order
from exact_tc import draw_disorder

WC, GCOLL, KAPPA, GDOWN, GPHI = 1.0, 0.8, 0.1, 0.02, 0.02


def support(key):
    _, _, es = key
    return frozenset(i for i, e in enumerate(es) if e != 0)


def build(M, Nc, sigma, seed, Fsafe):
    w = draw_disorder(M, WC, sigma, seed, "gauss")
    g = np.full(M, GCOLL / np.sqrt(M))
    return CumulantEngine(M, Nc, WC, w, g, kappa=KAPPA, gdown=GDOWN, gphi=GPHI, Fsafe=Fsafe)


def check_support_locality(eng, tol=1e-9):
    """(A) every nonzero C[k,j] must have |support(j) \\ support(k)| <= 1."""
    worst = 0
    n_nonzero = 0
    violations = []
    for a, k in enumerate(eng.track):
        sk = support(k)
        row = eng.C[a]
        for j, kj in enumerate(eng.full):
            c = row[j]
            if abs(c) > tol:
                n_nonzero += 1
                extra = len(support(kj) - sk)
                worst = max(worst, extra)
                if extra > 1:
                    violations.append((k, kj, abs(c), extra))
    return worst, n_nonzero, violations


def check_permutation(M, Nc, Fsafe, tol=1e-9):
    """(B) identical emitters: swapping sites 0<->1 must leave C invariant after relabeling monomials."""
    w = np.full(M, WC)                                  # identical emitters
    g = np.full(M, GCOLL / np.sqrt(M))
    eng = CumulantEngine(M, Nc, WC, w, g, kappa=KAPPA, gdown=GDOWN, gphi=GPHI, Fsafe=Fsafe)

    def swap01(key):
        p, q, es = key
        es = list(es)
        if M >= 2:
            es[0], es[1] = es[1], es[0]
        return (p, q, tuple(es))

    fidx = {kk: j for j, kk in enumerate(eng.full)}
    tidx = {kk: a for a, kk in enumerate(eng.track)}
    maxdiff = 0.0
    for a, k in enumerate(eng.track):
        ap = tidx[swap01(k)]
        for j, kj in enumerate(eng.full):
            jp = fidx[swap01(kj)]
            maxdiff = max(maxdiff, abs(eng.C[a, j] - eng.C[ap, jp]))
    return maxdiff


if __name__ == "__main__":
    print("STEP 1: structural locality of the validated dense cumulant EOM\n")
    print("(A) support locality: for all nonzero C[k,j], |support(j) minus support(k)| <= 1")
    for M, Nc, Fs in [(3, 10, 6), (4, 10, 6)]:
        eng = build(M, Nc, 0.15, 3, Fs)
        # confirm the engine itself is the validated one: gate at a quick exact state is implicit in prior runs;
        # here we only inspect its C. (The gate was already 1e-14 for these params.)
        worst, nnz, viol = check_support_locality(eng)
        status = "PASS" if worst <= 1 else "FAIL"
        print(f"   M={M}: nonzero C entries={nnz:5d}  max extra sites in any term = {worst}  -> {status}")
        if viol:
            for v in viol[:5]:
                print("      VIOLATION:", v)

    print("\n(B) permutation invariance (identical emitters), swap sites 0<->1:")
    for M, Nc, Fs in [(3, 10, 6), (4, 10, 6)]:
        d = check_permutation(M, Nc, Fs)
        status = "PASS" if d < 1e-9 else "FAIL"
        print(f"   M={M}: max |C - permuted C| = {d:.2e}  -> {status}")

    print("\nIf both PASS: the EOM is site-local (support grows by <=1) and permutation-structured, so it can be")
    print("assembled from cavity + <=3 site subspaces and scales to arbitrary M. Proceed to build that engine.")
