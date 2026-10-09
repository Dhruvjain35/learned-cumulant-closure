"""
Decisive test: measure the 3-POINT connected cumulants (the exact quantity a second-order
cumulant closure sets to zero) from the exact open Tavis-Cummings state, relative to the
2-point correlations. This is the real "room for a neural closure" signal.

If 3-point cumulants are tiny vs 2-point everywhere accessible, second-order cumulant is
already near-exact -> the neural-closure idea has little to fix. If they are substantial
(especially and growing with disorder), there is a genuine project.

Also: a Fock-cutoff convergence check, so no conclusion rests on truncation error.
"""
import os
import sys

import numpy as np
import qutip as qt
from exact_tc import build_ops, build_H, collapse_ops, draw_disorder, measure, bmf_indices

# Interpreter for any subprocess use: configurable, defaults to the running one. (Was a hard-coded local path;
# it is not used by the computation in this file.)
PY = os.environ.get("LCC_PYTHON", sys.executable)


def E(op, rho):
    return qt.expect(op, rho)


def conn3(A, B, C, rho):
    """3rd connected cumulant kappa3(A,B,C) = <ABC> - <A><BC> - <B><AC> - <C><AB> + 2<A><B><C>."""
    a, b, c = E(A, rho), E(B, rho), E(C, rho)
    ab, ac, bc = E(A * B, rho), E(A * C, rho), E(B * C, rho)
    abc = E(A * B * C, rho)
    return abc - a * bc - b * ac - c * ab + 2 * a * b * c


def cumulant_signal(rho, ops):
    """Representative 2-point connected and 3-point connected cumulant magnitudes."""
    a, sm, sp, sz, ee = ops["a"], ops["sm"], ops["sp"], ops["sz"], ops["ee"]
    M = ops["M"]
    # 2-point connected (what mean-field drops, cumulant keeps): <sp_i sm_j>_c and <a^dag sm_i>_c
    two = []
    for i in range(M):
        two.append(abs(E(a.dag() * sm[i], rho) - E(a.dag(), rho) * E(sm[i], rho)))
        for j in range(M):
            if i != j:
                two.append(abs(E(sp[i] * sm[j], rho) - E(sp[i], rho) * E(sm[j], rho)))
    two = np.array(two)
    # 3-point connected (what second-order cumulant drops = the neural-closure target):
    # representative distinct-index terms that appear in the TC moment hierarchy
    three = []
    # <a^dag sp_i sm_j>_c   (cavity + two emitters, i != j)
    for i in range(M):
        for j in range(M):
            if i != j:
                three.append(abs(conn3(a.dag(), sp[i], sm[j], rho)))
    # <sp_i sm_j sz_k>_c    (three distinct emitters)
    idx = list(range(M))
    for i in idx:
        for j in idx:
            for k in idx:
                if len({i, j, k}) == 3:
                    three.append(abs(conn3(sp[i], sm[j], sz[k], rho)))
    # <a^dag a sz_i>_c      (photon-number + emitter)
    for i in range(M):
        three.append(abs(conn3(a.dag(), a, sz[i], rho)))
    three = np.array(three) if three else np.array([0.0])
    two_scale = two.max() if two.size and two.max() > 0 else 1e-12
    return {
        "two_max": float(two.max()) if two.size else 0.0,
        "two_mean": float(two.mean()) if two.size else 0.0,
        "three_max": float(three.max()),
        "three_mean": float(three.mean()),
        "ratio_3over2_max": float(three.max() / two_scale),   # room above 2nd-order cumulant
    }


def run_point(M, Nc, wc, g0, sigma, seed, kappa, gpump, gphi, family="gauss"):
    ops = build_ops(M, Nc)
    w = draw_disorder(M, wc, sigma, seed, family)
    H = build_H(ops, wc, w, np.full(M, g0))
    c = collapse_ops(ops, kappa=kappa, gpump=gpump, gphi=gphi)
    rho = qt.steadystate(H, c)
    m = measure(rho, ops)
    nc = m["nc"]
    return {**bmf_indices(m), **cumulant_signal(rho, ops), "nc": nc, "Nc": Nc}


def convergence_check():
    print("\n[Convergence] pumped steady state M=4, raise Nc until stable")
    wc = 1.0; M = 4; g0 = 0.30 / np.sqrt(M); kappa, gpump, gphi = 0.10, 0.05, 0.02
    for Nc in [3, 4, 5, 6]:
        r = run_point(M, Nc, wc, g0, 0.0, 0, kappa, gpump, gphi)
        flag = "" if r["nc"] < 0.4 * Nc else "  <-- nc near cutoff"
        print(f"   Nc={Nc}: nc={r['nc']:.4f}  C_ee/pe={r['C_ee_over_pe']:.4f}  "
              f"two_max={r['two_max']:.4f}  three_max={r['three_max']:.5f}  3/2={r['ratio_3over2_max']:.4f}{flag}")


def regime_scan():
    print("\n[Regime scan] 3-point-cumulant (closure target) vs 2-point, converged Nc, disorder sweep")
    wc = 1.0
    regimes = [
        ("weak-pump",   dict(gcoll=0.30, kappa=0.10, gpump=0.05, gphi=0.02)),
        ("strong-coup", dict(gcoll=0.80, kappa=0.10, gpump=0.05, gphi=0.02)),
        ("hard-pump",   dict(gcoll=0.50, kappa=0.10, gpump=0.30, gphi=0.05)),
    ]
    M = 3
    Nc = 6   # generous cutoff; convergence_check justifies it
    for name, p in regimes:
        g0 = p["gcoll"] / np.sqrt(M)
        print(f"\n   regime={name}  gcoll={p['gcoll']} kappa={p['kappa']} gpump={p['gpump']} gphi={p['gphi']}  (M={M}, Nc={Nc})")
        for sigma in [0.0, 0.05, 0.15, 0.30]:
            vals = [run_point(M, Nc, wc, g0, sigma, s, p["kappa"], p["gpump"], p["gphi"]) for s in range(4)]
            twom = np.mean([v["two_max"] for v in vals])
            thrm = np.mean([v["three_max"] for v in vals])
            ratio = np.mean([v["ratio_3over2_max"] for v in vals])
            ncm = np.mean([v["nc"] for v in vals])
            print(f"     sigma={sigma:.2f}: two_max={twom:.4f}  three_max(closure target)={thrm:.5f}  "
                  f"3/2 ratio={ratio:.4f}  nc={ncm:.3f}")


if __name__ == "__main__":
    print("=" * 78)
    print("DECISIVE TEST: 3-point connected cumulants (the neural-closure target) in open TC")
    print("=" * 78)
    convergence_check()
    regime_scan()
    print("\ninterpretation key:")
    print("  two_max   = largest 2-point connected correlation (mean-field error scale)")
    print("  three_max = largest 3-point connected cumulant     (2nd-order-CUMULANT error scale = the closure target)")
    print("  3/2 ratio = three_max/two_max: if << 1, 2nd-order cumulant already near-exact (little to learn)")
