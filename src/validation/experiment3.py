"""
Clean, trustworthy version. The photon-involving 3-point cumulants were Fock-truncation
contaminated. Emitter (spin-1/2) operators are BOUNDED, so pure-spin cumulants are
truncation-safe once the STATE is converged. We:
  1. separate cumulants into pure-spin (safe) vs cavity-involving (flagged),
  2. verify state convergence by comparing Nc vs Nc+2 on the spin cumulants,
  3. report the pure-spin 2-point and 3-point connected cumulants (the trustworthy signal),
     the ratio c3/c2 (room above 2nd-order cumulant), and how they move with M and disorder.

Interpretation: spin operators are O(1). A pure-spin 3-point cumulant of, say, 0.05 means the
term a 2nd-order cumulant closure drops is ~5% of the spin scale -> real room for a closure.
If it is ~1e-3, second-order cumulant is already near-exact -> the neural-closure idea is dead.
"""
import numpy as np
import qutip as qt
from exact_tc import build_ops, build_H, collapse_ops, draw_disorder


def E(op, rho):
    return qt.expect(op, rho)


def conn2(A, B, rho):
    return E(A * B, rho) - E(A, rho) * E(B, rho)


def conn3(A, B, C, rho):
    a, b, c = E(A, rho), E(B, rho), E(C, rho)
    ab, ac, bc = E(A * B, rho), E(A * C, rho), E(B * C, rho)
    return E(A * B * C, rho) - a * bc - b * ac - c * ab + 2 * a * b * c


def spin_cumulants(rho, ops):
    """Pure-spin (bounded, truncation-safe) 2- and 3-point connected cumulants."""
    sm, sp, sz = ops["sm"], ops["sp"], ops["sz"]
    M = ops["M"]
    two = [abs(conn2(sp[i], sm[j], rho)) for i in range(M) for j in range(M) if i != j]
    three = []
    for i in range(M):
        for j in range(M):
            for k in range(M):
                if len({i, j, k}) == 3:
                    three.append(abs(conn3(sp[i], sm[j], sz[k], rho)))
    two = np.array(two) if two else np.array([0.0])
    three = np.array(three) if three else np.array([np.nan])   # M<3 has no distinct triple
    return {"c2_max": float(two.max()), "c2_mean": float(two.mean()),
            "c3_max": float(np.nanmax(three)) if np.isfinite(three).any() else float("nan"),
            "c3_mean": float(np.nanmean(three)) if np.isfinite(three).any() else float("nan")}


def solve(M, Nc, wc, g0, sigma, seed, kappa, gpump, gphi):
    ops = build_ops(M, Nc)
    w = draw_disorder(M, wc, sigma, seed, "gauss")
    H = build_H(ops, wc, w, np.full(M, g0))
    c = collapse_ops(ops, kappa=kappa, gpump=gpump, gphi=gphi)
    rho = qt.steadystate(H, c)
    nc = E(ops["a"].dag() * ops["a"], rho).real
    return {**spin_cumulants(rho, ops), "nc": nc}


def state_convergence():
    print("\n[State convergence] pure-spin cumulants, Nc vs Nc+2 (must agree)")
    wc = 1.0; M = 3; gcoll = 0.8; g0 = gcoll / np.sqrt(M); kappa, gpump, gphi = 0.10, 0.05, 0.02
    for Nc in [5, 7, 9]:
        r = solve(M, Nc, wc, g0, 0.15, 1, kappa, gpump, gphi)
        print(f"   Nc={Nc}: nc={r['nc']:.4f}  c2_max={r['c2_max']:.5f}  c3_max={r['c3_max']:.5f}")


def signal_scan():
    print("\n[Signal] pure-spin closure target c3 vs c2, sweeping M and disorder (strong-coupling regime)")
    wc = 1.0; gcoll = 0.8; kappa, gpump, gphi = 0.10, 0.05, 0.02
    Nc = 8
    rows = []
    for M in [3, 4]:
        g0 = gcoll / np.sqrt(M)
        print(f"\n   M={M}  (gcoll={gcoll}, kappa={kappa}, gpump={gpump}, gphi={gphi}, Nc={Nc})")
        for sigma in [0.0, 0.05, 0.15, 0.30, 0.50]:
            vals = [solve(M, Nc, wc, g0, sigma, s, kappa, gpump, gphi) for s in range(5)]
            c2 = np.mean([v["c2_max"] for v in vals]); c2s = np.std([v["c2_max"] for v in vals])
            c3 = np.mean([v["c3_max"] for v in vals]); c3s = np.std([v["c3_max"] for v in vals])
            nc = np.mean([v["nc"] for v in vals])
            ratio = c3 / c2 if c2 > 1e-12 else float("nan")
            rows.append((M, sigma, c2, c3, ratio, nc))
            print(f"     sigma={sigma:.2f}: c2_max={c2:.5f}+-{c2s:.5f}  c3_max(closure target)={c3:.5f}+-{c3s:.5f}  "
                  f"c3/c2={ratio:.3f}  nc={nc:.2f}")
    return rows


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for M in [3, 4]:
        sub = [r for r in rows if r[0] == M]
        sig = [r[1] for r in sub]
        ax[0].plot(sig, [r[3] for r in sub], "o-", label=f"M={M}")
        ax[1].plot(sig, [r[2] for r in sub], "s-", label=f"c2 M={M}")
        ax[1].plot(sig, [r[3] for r in sub], "o--", label=f"c3 M={M}")
    ax[0].set_xlabel("disorder sigma"); ax[0].set_ylabel("c3_max (3-pt spin cumulant = closure target)")
    ax[0].set_title("Does the closure target grow with disorder?"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].set_xlabel("disorder sigma"); ax[1].set_ylabel("connected spin cumulant")
    ax[1].set_title("2-point (mean-field err) vs 3-point (cumulant err)"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig("results/closure_signal.png", dpi=130)
    print("\nsaved results/closure_signal.png")


if __name__ == "__main__":
    print("=" * 80)
    print("TRUSTWORTHY TEST: pure-spin connected cumulants in open disordered Tavis-Cummings")
    print("=" * 80)
    state_convergence()
    rows = signal_scan()
    plot(rows)
