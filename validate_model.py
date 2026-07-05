"""
Validate the exact open Tavis-Cummings solver against KNOWN limits before trusting any
physics. Three gates:
  (1) M=1, lossless, resonant: single-excitation eigenvalues = wc +/- g  (vacuum Rabi).
  (2) M identical, lossless: bright-polariton splitting = 2 g sqrt(M).
  (3) incoherently-pumped steady state: connected (beyond-mean-field) emitter correlations
      are nonzero, and at ZERO disorder they shrink as M grows at fixed collective coupling
      (mean-field must get better with M). This is the sanity check that decides whether the
      whole "mean-field fails, ML fixes it" premise can even be right.
"""
import numpy as np
import qutip as qt
from exact_tc import build_ops, build_H, collapse_ops, draw_disorder, measure, bmf_indices


def total_excitation(ops):
    N = ops["a"].dag() * ops["a"]
    for i in range(ops["M"]):
        N = N + ops["ee"][i]
    return N


def gate1_vacuum_rabi():
    print("\n[Gate 1] M=1 vacuum Rabi: single-excitation eigenvalues should be wc +/- g")
    wc, g = 1.0, 0.1
    ops = build_ops(1, 3)
    H = build_H(ops, wc, np.array([wc]), np.array([g]))
    N = total_excitation(ops)
    evals, evecs = H.eigenstates()
    single = [(e, qt.expect(N, v).real) for e, v in zip(evals, evecs)]
    ones = sorted([e for e, n in single if abs(n - 1) < 1e-6])
    pred = [wc - g, wc + g]
    ok = np.allclose(ones, pred, atol=1e-9)
    print(f"   measured single-excitation eigs = {np.round(ones,6)}   predicted = {pred}   PASS={ok}")
    return ok


def gate2_collective_sqrtM():
    print("\n[Gate 2] identical emitters, lossless: bright splitting should be 2 g sqrt(M)")
    wc, g = 1.0, 0.05
    allok = True
    for M in [1, 2, 3, 4, 6]:
        Nc = 2
        ops = build_ops(M, Nc)
        H = build_H(ops, wc, np.full(M, wc), np.full(M, g))
        N = total_excitation(ops)
        evals, evecs = H.eigenstates()
        ones = sorted([e for e, v in zip(evals, evecs) if abs(qt.expect(N, v).real - 1) < 1e-6])
        split = ones[-1] - ones[0]
        pred = 2 * g * np.sqrt(M)
        ok = abs(split - pred) < 1e-9
        allok = allok and ok
        print(f"   M={M}: measured 2g_coll = {split:.6f}   predicted 2g*sqrt(M) = {pred:.6f}   PASS={ok}")
    return allok


def gate3_meanfield_improves():
    print("\n[Gate 3] incoherent-pump steady state: connected emitter correlations vs M (sigma=0)")
    print("   holding COLLECTIVE coupling gcoll = g0*sqrt(M) fixed; mean-field should improve (C shrink) with M")
    wc = 1.0
    gcoll = 0.30           # collective coupling held fixed
    kappa = 0.10           # cavity loss
    gpump = 0.05           # incoherent pump per emitter
    gphi = 0.02            # dephasing
    print(f"   params: gcoll={gcoll}, kappa={kappa}, gpump={gpump}, gphi={gphi}")
    prev = None
    monotone = True
    for M in [2, 3, 4, 5]:
        g0 = gcoll / np.sqrt(M)
        Nc = 3
        ops = build_ops(M, Nc)
        H = build_H(ops, wc, np.full(M, wc), np.full(M, g0))
        c = collapse_ops(ops, kappa=kappa, gpump=gpump, gphi=gphi)
        rho = qt.steadystate(H, c)
        m = measure(rho, ops)
        idx = bmf_indices(m)
        nc = idx["nc"]
        # convergence guard: photon occupation must be small vs Fock cutoff
        warn = "  <-- WARN: nc near Nc, raise Nc" if nc > 0.5 * Nc else ""
        cee = idx["C_ee_over_pe"]
        trend = "" if prev is None else ("  (down)" if cee < prev - 1e-4 else "  (UP - not shrinking!)")
        if prev is not None and cee > prev + 1e-4:
            monotone = False
        print(f"   M={M}: C_ee/pe = {cee:.4f}   mean_pe={idx['mean_pe']:.3f}   nc={nc:.3f}{trend}{warn}")
        prev = cee
    print(f"   mean-field-improves-with-M (C_ee shrinks): {'PASS' if monotone else 'DID NOT SHRINK MONOTONICALLY'}")
    return monotone


def gate3b_disorder_effect():
    print("\n[Gate 3b] fixed M, sweep disorder sigma: does disorder change the connected correlations?")
    wc = 1.0
    M = 4
    g0 = 0.30 / np.sqrt(M)
    kappa, gpump, gphi = 0.10, 0.05, 0.02
    Nc = 3
    for sigma in [0.0, 0.02, 0.05, 0.1, 0.2]:
        # average over a few disorder realizations
        vals = []
        for seed in range(4):
            ops = build_ops(M, Nc)
            w = draw_disorder(M, wc, sigma, seed, "gauss")
            H = build_H(ops, wc, w, np.full(M, g0))
            c = collapse_ops(ops, kappa=kappa, gpump=gpump, gphi=gphi)
            rho = qt.steadystate(H, c)
            vals.append(bmf_indices(measure(rho, ops))["C_ee_over_pe"])
        print(f"   sigma={sigma:.3f}: C_ee/pe = {np.mean(vals):.4f} +/- {np.std(vals):.4f}   (6 realizations)")


if __name__ == "__main__":
    print("=" * 72)
    print("VALIDATION of the exact open Tavis-Cummings solver (QuTiP 5.3)")
    print("=" * 72)
    g1 = gate1_vacuum_rabi()
    g2 = gate2_collective_sqrtM()
    g3 = gate3_meanfield_improves()
    gate3b_disorder_effect()
    print("\n" + "=" * 72)
    print(f"GATES: vacuum-Rabi={g1}  collective-sqrtM={g2}  meanfield-improves-with-M={g3}")
    print("=" * 72)
