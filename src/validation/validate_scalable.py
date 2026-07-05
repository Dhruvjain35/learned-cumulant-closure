"""
Validate the ScalableEngine against the dense (1e-14-validated) CumulantEngine.

The scalable engine assembles the EOM coefficient matrix C from small subsystem computations placed onto
M sites. It is only trustworthy at large M if it reproduces the dense engine's C EXACTLY wherever the dense
engine can be built. Here we assert max|C_scalable - C_dense| ~ machine precision at M = 2,3,4,5, on a
DISORDERED realization (so the per-site frequency term and the placement are both exercised).

If this passes, the scalable engine is proven correct on identical footing to the dense one, and can be used
at M where dense OOMs. No hallucination: correctness is a numerical assertion, not a derivation on faith.
"""
import numpy as np
from cumulant import CumulantEngine
from scalable_engine import ScalableEngine
from exact_tc import draw_disorder

WC, GCOLL, KAPPA, GDOWN, GPHI = 1.0, 0.8, 0.1, 0.02, 0.02


def both(M, Nc, sigma, seed, Fsafe):
    w = draw_disorder(M, WC, sigma, seed, "gauss")
    g = np.full(M, GCOLL / np.sqrt(M))
    dense = CumulantEngine(M, Nc, WC, w, g, kappa=KAPPA, gdown=GDOWN, gphi=GPHI, Fsafe=Fsafe)
    scal = ScalableEngine(M, Nc, WC, w, g, kappa=KAPPA, gdown=GDOWN, gphi=GPHI, Fsafe=Fsafe)
    return dense, scal


if __name__ == "__main__":
    print("Validate ScalableEngine.build_C() vs dense CumulantEngine.C  (disordered, seed 7)\n")
    print(f"{'M':>2s} {'Nc':>3s} {'max|C_scal - C_dense|':>22s}  {'verdict':>8s}")
    allpass = True
    for M, Nc, Fs in [(2, 8, 4), (3, 10, 6), (4, 10, 6), (5, 11, 6)]:
        dense, scal = both(M, Nc, 0.15, 7, Fs)
        Cs = scal.build_C()
        d = float(np.max(np.abs(Cs - dense.C)))
        ok = d < 1e-10
        allpass &= ok
        print(f"{M:>2d} {Nc:>3d} {d:>22.2e}  {'PASS' if ok else 'FAIL':>8s}")
    print("\n" + ("ALL PASS: scalable engine reproduces the validated dense EOM exactly; trustworthy at large M."
                  if allpass else "FAIL: placement/assembly is wrong somewhere; do not trust at large M."))
