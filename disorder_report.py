"""
Rigorous read of the disorder sweep (results/disorder_sweep.npz): mean +/- SEM across the 5 held-out draws,
signal-vs-noise per quantity, sigma vs physical scales, and the pre-registered kill/survive call. No spin.
Pre-registered SURVIVE = non-monotonic intermediate-sigma structure in the kappa3 channels AND the 2nd-order
error, tied to a burst-observable change. Anything monotone = the expected, known behavior = KILL the novelty.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NSEED = 5
GC, KA, GP, GDOWN = 0.70, 0.03, 0.02, 0.02
GPERP = GDOWN / 2 + GP
SIGMAS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.45]
QTS = [("Fpeak", "burst peak flux F_peak"), ("tpeak", "burst delay t_peak"),
       ("Aaz", "kappa3 photon-number-inversion (dominant)"),
       ("a+z", "kappa3 photon-polarization-inversion (secondary)"),
       ("total_k3", "kappa3 total"), ("perr", "2nd-order F_peak error")]


def monotonic(ys):
    d = np.diff(ys)
    return np.all(d >= -1e-9) or np.all(d <= 1e-9)


def analyze():
    res = np.load("results/disorder_sweep.npz", allow_pickle=True)["res"].item()
    Ms = sorted({M for (M, _) in res})
    print(f"physical scales: g_c={GC}, kappa={KA}, gamma_perp={GPERP:.3f}, gdown={GDOWN}; "
          f"sigma range {SIGMAS[0]}..{SIGMAS[-1]}  (sigma/g_c up to {SIGMAS[-1]/GC:.2f}, "
          f"sigma/kappa up to {SIGMAS[-1]/KA:.0f})\n")
    verdict = {}
    for q, name in QTS:
        print(f"[{name}]")
        for M in Ms:
            sgs = [sg for sg in SIGMAS if (M, sg) in res]
            mean = np.array([res[(M, sg)][q][0] for sg in sgs])
            sem = np.array([res[(M, sg)][q][1] / np.sqrt(NSEED) for sg in sgs])
            signal = float(np.nanmax(mean) - np.nanmin(mean))
            noise = float(np.nanmax(sem)) if np.nanmax(sem) > 0 else 1e-9
            rel = signal / max(abs(np.nanmean(mean)), 1e-12)
            argmax = int(np.nanargmax(mean))
            shape = "MONOTONIC" if monotonic(mean) else f"NON-MONOTONIC (peak sigma={sgs[argmax]:.2f})"
            print(f"   M={M}: range={signal:.4f} ({rel*100:.0f}% rel), max SEM={noise:.4f}, "
                  f"signal/noise={signal/noise:.1f}, {shape}")
            verdict[(q, M)] = (monotonic(mean), signal / noise, rel)
        print()
    # pre-registered survive check: need NON-MONOTONIC in a kappa3 channel AND in 2nd-ord err, signal/noise>3
    surv = any((not verdict[(q, M)][0]) and verdict[(q, M)][1] > 3
               for q in ("Aaz", "a+z") for M in Ms) and \
           any((not verdict[("perr", M)][0]) and verdict[("perr", M)][1] > 3 for M in Ms)
    print("=== PRE-REGISTERED CALL ===")
    print(f"  survive condition (non-monotonic intermediate-sigma structure in a kappa3 channel AND in the")
    print(f"  2nd-order error, signal/noise>3): {'MET -> worth Phase II' if surv else 'NOT MET -> KILL the novelty claim'}")
    return res, Ms


def figure(res, Ms):
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25})
    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    for a, (q, name) in zip(ax.ravel(), QTS):
        for M in Ms:
            sgs = [sg for sg in SIGMAS if (M, sg) in res]
            mean = [res[(M, sg)][q][0] for sg in sgs]
            sem = [res[(M, sg)][q][1] / np.sqrt(NSEED) for sg in sgs]
            a.errorbar(sgs, mean, yerr=sem, marker="o", ms=3, capsize=2, label=f"M={M}")
        a.set_title(name, fontsize=8.5)
        a.set_xlabel("spectral disorder σ")
        for xs, lab in [(KA, "κ"), (GPERP, "γ⊥")]:
            a.axvline(xs, color="#bbb", ls=":", lw=0.8)
        a.legend(fontsize=7)
    fig.suptitle("Disorder sweep (exact, fixed G=g_c): burst observables, connected κ3 channels, 2nd-order error "
                 "(mean ± SEM, 5 draws)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("results/disorder_sweep_fig.png", dpi=140)
    print("saved results/disorder_sweep_fig.png")


if __name__ == "__main__":
    res, Ms = analyze()
    figure(res, Ms)
