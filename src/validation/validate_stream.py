"""
Validate the streaming deriv (used at large M) against the assembled matrix build_C(), and confirm the
sparse monomial enumerator matches the dense one. These are pure consistency checks (no dense QuTiP engine),
so they are fast and can run at small M where build_C() is available.

  (1) enum_sparse(M,K) yields the same SET as enum_monos(M,K)  -> sparse enumeration loses nothing.
  (2) deriv_stream(lookup) == build_C() @ m_full  for random m_full, treating every order<=3 monomial as an
      independent value (lookup by index). This isolates the placement/streaming logic from the closure:
      if it matches, the streaming EOM is the same linear map as the validated matrix.
"""
import numpy as np
from cumulant import enum_monos
from scalable_engine import ScalableEngine, enum_sparse

WC, GCOLL, KAPPA, GDOWN, GPHI = 1.0, 0.8, 0.1, 0.02, 0.02


def make(M, Nc, Fs, seed=7, sigma=0.15):
    rng = np.random.default_rng(seed)
    w = WC + sigma * rng.standard_normal(M)
    g = np.full(M, GCOLL / np.sqrt(M))
    return ScalableEngine(M, Nc, WC, w, g, kappa=KAPPA, gdown=GDOWN, gphi=GPHI, Fsafe=Fs)


if __name__ == "__main__":
    print("(1) enum_sparse == enum_monos (as sets):")
    for M in [2, 3, 4, 5, 6]:
        for K in [2, 3]:
            a = set(enum_monos(M, K))
            b = set(enum_sparse(M, K))
            print(f"    M={M} K={K}: match={a == b}  (|set|={len(a)})")

    print("\n(2) deriv_stream == build_C() @ m_full  (random moments, lookup by index):")
    allok = True
    for M, Nc, Fs in [(3, 10, 6), (4, 10, 6)]:
        eng = make(M, Nc, Fs)
        eng.precompute()
        C = eng.build_C()
        rng = np.random.default_rng(0)
        mfull = rng.standard_normal(len(eng.full)) + 1j * rng.standard_normal(len(eng.full))
        d_mat = C @ mfull
        d_stream = eng.deriv_stream(lambda gk: mfull[eng.idx[gk]])
        err = float(np.max(np.abs(d_mat - d_stream)))
        ok = err < 1e-12
        allok &= ok
        print(f"    M={M}: max|stream - matrix| = {err:.2e}  {'PASS' if ok else 'FAIL'}")
    print("\n" + ("PASS: streaming EOM equals the validated matrix map; safe to use at large M."
                  if allok else "FAIL: streaming placement disagrees with build_C()."))
