"""
Validate deriv_compiled (the fast vectorized EOM used for large-M runs) against the assembled matrix
build_C(), and measure its speed as M grows (to find the feasible ceiling). Correctness first, then timing.

Consistency check: with order-3 moments treated as INDEPENDENT values v3 (i.e. bypassing the cumulant
factorization, exactly what build_C @ m_full does), deriv_compiled must equal build_C() @ m_full.
"""
import time
import numpy as np
from scalable_engine import ScalableEngine

WC, GCOLL, KAPPA, GDOWN, GPHI = 1.0, 0.8, 0.1, 0.02, 0.02


def make(M, Nc, Fs, seed=7, sigma=0.15):
    rng = np.random.default_rng(seed)
    w = WC + sigma * rng.standard_normal(M)
    g = np.full(M, GCOLL / np.sqrt(M))
    return ScalableEngine(M, Nc, WC, w, g, kappa=KAPPA, gdown=GDOWN, gphi=GPHI, Fsafe=Fs)


if __name__ == "__main__":
    print("(1) deriv_compiled == build_C() @ m_full  (order-3 treated as independent):")
    allok = True
    for M, Nc, Fs in [(3, 10, 6), (4, 10, 6), (5, 11, 6)]:
        eng = make(M, Nc, Fs).compile()
        C = eng.build_C()
        rng = np.random.default_rng(1)
        mfull = rng.standard_normal(len(eng.full)) + 1j * rng.standard_normal(len(eng.full))
        m_tr = np.array([mfull[eng.idx[k]] for k in eng.track])
        v3 = np.array([mfull[eng.idx[gk]] for gk in eng.o3_keys]) - eng.v3_factorized(m_tr)
        d_comp = eng.deriv_compiled(m_tr, v3)          # v3 offset makes the internal factorization cancel
        d_mat = (C @ mfull)
        err = float(np.max(np.abs(d_comp - d_mat)))
        ok = err < 1e-11
        allok &= ok
        print(f"    M={M}: max|compiled - matrix| = {err:.2e}  {'PASS' if ok else 'FAIL'}")
    print("    " + ("PASS" if allok else "FAIL") + "\n")

    print("(2) build + compile + deriv speed as M grows (no full C; sparse/vectorized):")
    print(f"    {'M':>3s} {'tracked':>8s} {'o3_keys':>8s} {'compile_s':>10s} {'ms/deriv':>9s}")
    for M in [8, 12, 16, 20, 30, 40]:
        t0 = time.time()
        eng = make(M, 12, 6).compile()
        tc = time.time() - t0
        m = np.zeros(len(eng.track), complex)
        rng = np.random.default_rng(0)
        m[:] = 0.1 * (rng.standard_normal(len(eng.track)) + 1j * rng.standard_normal(len(eng.track)))
        kap = np.zeros(len(eng.o3_keys), complex)
        N = 50
        t0 = time.time()
        for _ in range(N):
            eng.deriv_compiled(m, kap)
        msd = 1e3 * (time.time() - t0) / N
        print(f"    {M:>3d} {len(eng.track):>8d} {len(eng.o3_keys):>8d} {tc:>10.1f} {msd:>9.2f}")
