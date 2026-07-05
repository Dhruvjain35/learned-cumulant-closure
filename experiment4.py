"""
Decisive high-signal regime: TRANSIENT SUPERRADIANCE. All M emitters start excited, cavity in
vacuum; they collectively dump energy into the lossy cavity. Beyond-cumulant correlations are
known to be largest during the superradiant burst. We track the pure-spin connected cumulants
(truncation-safe) over time, take the PEAK, and see how the closure target (3-point cumulant)
scales with disorder and M. Fock cutoff convergence checked.

If the peak 3-point spin cumulant is still ~1% here, the neural-closure idea is weak everywhere
accessible. If it is tens of percent and disorder-enhanced, there is a real project.
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


def peak_cumulants(M, Nc, wc, g0, sigma, seed, kappa, gdown, gphi, tmax, nt):
    ops = build_ops(M, Nc)
    sm, sp, sz, a = ops["sm"], ops["sp"], ops["sz"], ops["a"]
    w = draw_disorder(M, wc, sigma, seed, "gauss")
    H = build_H(ops, wc, w, np.full(M, g0))
    c = collapse_ops(ops, kappa=kappa, gdown=gdown, gphi=gphi)
    psi0 = qt.tensor(qt.basis(Nc, 0), *[qt.basis(2, 0) for _ in range(M)])   # cavity vacuum, all emitters excited
    ts = np.linspace(0, tmax, nt)
    res = qt.mesolve(H, psi0, ts, c_ops=c, options={"progress_bar": False})
    c2t, c3t, nct = [], [], []
    trip = [(i, j, k) for i in range(M) for j in range(M) for k in range(M) if len({i, j, k}) == 3]
    pair = [(i, j) for i in range(M) for j in range(M) if i != j]
    for rho in res.states:
        c2t.append(max(abs(conn2(sp[i], sm[j], rho)) for (i, j) in pair))
        c3t.append(max((abs(conn3(sp[i], sm[j], sz[k], rho)) for (i, j, k) in trip), default=np.nan))
        nct.append(E(a.dag() * a, rho).real)
    c2t, c3t, nct = np.array(c2t), np.array(c3t), np.array(nct)
    return {"c2_peak": float(np.nanmax(c2t)), "c3_peak": float(np.nanmax(c3t)),
            "nc_peak": float(np.nanmax(nct)), "c3_at_c3peak_over_c2": float(np.nanmax(c3t) / (np.nanmax(c2t) + 1e-12))}


def conv_check():
    print("\n[Fock convergence] transient superradiance M=3, raise Nc")
    for Nc in [4, 6, 8]:
        r = peak_cumulants(3, Nc, 1.0, 0.8 / np.sqrt(3), 0.15, 1, kappa=0.1, gdown=0.02, gphi=0.02, tmax=40, nt=200)
        print(f"   Nc={Nc}: nc_peak={r['nc_peak']:.3f}  c2_peak={r['c2_peak']:.4f}  c3_peak={r['c3_peak']:.4f}")


def scan():
    print("\n[Superradiant burst] peak pure-spin cumulants vs disorder and M")
    wc, gcoll, kappa, gdown, gphi = 1.0, 0.8, 0.1, 0.02, 0.02
    rows = []
    for M in [3, 4]:
        Nc = 8
        g0 = gcoll / np.sqrt(M)
        print(f"\n   M={M} (gcoll={gcoll}, kappa={kappa}, Nc={Nc})")
        for sigma in [0.0, 0.05, 0.15, 0.30, 0.50]:
            vals = [peak_cumulants(M, Nc, wc, g0, sigma, s, kappa, gdown, gphi, tmax=40, nt=200) for s in range(4)]
            c2 = np.mean([v["c2_peak"] for v in vals]); c3 = np.mean([v["c3_peak"] for v in vals])
            c3s = np.std([v["c3_peak"] for v in vals]); nc = np.mean([v["nc_peak"] for v in vals])
            rows.append((M, sigma, c2, c3, c3 / c2 if c2 > 1e-9 else np.nan, nc))
            print(f"     sigma={sigma:.2f}: c2_peak={c2:.4f}  c3_peak(closure target)={c3:.4f}+-{c3s:.4f}  "
                  f"c3/c2={c3/c2 if c2>1e-9 else float('nan'):.3f}  nc_peak={nc:.2f}")
    return rows


if __name__ == "__main__":
    print("=" * 80)
    print("TRANSIENT SUPERRADIANCE: peak beyond-cumulant signal (the high-signal regime)")
    print("=" * 80)
    conv_check()
    scan()
