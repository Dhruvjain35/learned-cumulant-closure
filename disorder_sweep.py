"""
CHEAP EXACT KILL TEST (candidate Phase II, honest framing): does spectral disorder sigma change the two
Result-25 connected third-order light-matter correlation channels of the superradiant burst, the burst
observables, and the SECOND-ORDER closure's error -- in a way that is non-trivial (ideally non-monotonic,
peaking at intermediate sigma) rather than just tracking the mean?

NO V7 here (exact + plain 2nd-order only). NO non-Gaussianity/Wigner/metrology language: kappa3 = connected
third-order correlation, nothing more. Fixed g_c => G^2 = g_c^2 fixed automatically as sigma varies.

KILL: kappa3 channels AND 2nd-order F_peak error flat/trivially-monotone in sigma, no burst-observable
structure -> null, a plot not a contribution. SURVIVES: non-trivial intermediate-sigma structure tied to a
burst-observable change, consistent across held-out draws.
"""
import sys
import numpy as np

from pilot_eval import build_cell, m0_product, run_from, WC
from local_run import observables, stable_ok
from mechanism import exact_moment_traj, kappa3_of, build_families, fam_label
from burst_pred import burst_obs, NC_REF

GC, KA, GP = 0.70, 0.03, 0.02
SIGMAS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.45]
MS = [3, 4]
SEEDS = [60, 61, 62, 63, 64]
FLUX_FRAC = 0.5
CHANNELS = {"Aaz": "photon-number-inversion", "a+z": "photon-polarization-inversion"}
SMOKE = "--smoke" in sys.argv
if SMOKE:
    SIGMAS, MS, SEEDS = [0.0, 0.15, 0.30], [3], [60]


def channel_mag(kap_ex, win, idx):
    sub = kap_ex[np.ix_(win, idx)]
    return float(np.mean(np.sqrt(np.sum(np.abs(sub) ** 2, axis=1))))


def run_one(M, sigma, seed):
    eng, w, g = build_cell(M, GC, KA, GP, sigma, seed)
    fam2idx = build_families(eng)
    lab2fam = {fam_label(f): f for f in fam2idx}
    ts, m2ex, raw3ex, nex, frac = exact_moment_traj(M, w, g, KA, GP, eng.track, eng.o3_keys, NC_REF)
    if frac >= 0.45:
        return None
    kap_ex = np.array([kappa3_of(eng, m2ex[t], raw3ex[t]) for t in range(len(ts))])
    flux = KA * nex
    win = np.where(flux >= FLUX_FRAC * flux.max())[0]
    b = burst_obs(ts, nex, KA)
    # two channels' connected-correlation magnitude in the burst window (0 if channel absent at this M)
    mags = {}
    for lab in CHANNELS:
        mags[lab] = channel_mag(kap_ex, win, fam2idx[lab2fam[lab]]) if lab in lab2fam else 0.0
    total_k3 = channel_mag(kap_ex, win, np.arange(kap_ex.shape[1]))
    # plain second-order closure error on F_peak
    _, tp = run_from(eng, m0_product(eng, 1.0), None)
    perr = (abs(burst_obs(ts, observables(eng, tp)[0], KA)["Fpeak"] - b["Fpeak"]) / max(b["Fpeak"], 1e-12)
            if stable_ok(tp) else np.nan)
    return dict(Fpeak=b["Fpeak"], tpeak=b["tpeak"], total_k3=total_k3, perr=perr, **mags)


if __name__ == "__main__":
    print(f"DISORDER SWEEP {'(SMOKE) ' if SMOKE else ''}(exact + plain 2nd-order); g_c={GC} kappa={KA} gphi={GP}; "
          f"M={MS}; seeds={SEEDS}\n  channels: {CHANNELS}\n")
    res = {}
    for M in MS:
        print(f"=== M={M} ===")
        print(f"  {'sigma':>5s} | {'k3(Aaz)':>9s} {'k3(a+z)':>9s} {'k3_total':>9s} | "
              f"{'F_peak':>8s} {'t_peak':>7s} | {'2nd-ord F_peak err':>18s}   (mean +/- sd over seeds)")
        for sg in SIGMAS:
            rows = [run_one(M, sg, sd) for sd in SEEDS]
            rows = [r for r in rows if r is not None]
            if not rows:
                print(f"  {sg:>5.2f} | all dropped (Fock guard)"); continue
            agg = {k: (np.nanmean([r[k] for r in rows]), np.nanstd([r[k] for r in rows]))
                   for k in ("Aaz", "a+z", "total_k3", "Fpeak", "tpeak", "perr")}
            res[(M, sg)] = agg
            print(f"  {sg:>5.2f} | {agg['Aaz'][0]:>9.4f} {agg['a+z'][0]:>9.4f} {agg['total_k3'][0]:>9.4f} | "
                  f"{agg['Fpeak'][0]:>8.4f} {agg['tpeak'][0]:>7.2f} | "
                  f"{agg['perr'][0]:>8.3f} +/- {agg['perr'][1]:<7.3f}", flush=True)
        # trivial-monotonicity / structure check on the primary channel + 2nd-order error
        sk = [res[(M, sg)]["Aaz"][0] for sg in SIGMAS if (M, sg) in res]
        pe = [res[(M, sg)]["perr"][0] for sg in SIGMAS if (M, sg) in res]
        sgs = [sg for sg in SIGMAS if (M, sg) in res]
        if len(sk) >= 3:
            amax = int(np.nanargmax(sk))
            nonmono_k3 = 0 < amax < len(sk) - 1
            amax_pe = int(np.nanargmax(pe))
            nonmono_pe = 0 < amax_pe < len(pe) - 1
            rng_k3 = (np.nanmax(sk) - np.nanmin(sk)) / max(abs(np.nanmean(sk)), 1e-9)
            rng_pe = (np.nanmax(pe) - np.nanmin(pe)) / max(abs(np.nanmean(pe)), 1e-9)
            print(f"  -> k3(Aaz): peak at sigma={sgs[amax]:.2f} ({'NON-MONOTONIC' if nonmono_k3 else 'endpoint/monotone'}), "
                  f"rel.range={rng_k3:.2f}; 2nd-ord err: peak at sigma={sgs[amax_pe]:.2f} "
                  f"({'NON-MONOTONIC' if nonmono_pe else 'endpoint/monotone'}), rel.range={rng_pe:.2f}")
        print()
    np.savez_compressed("results/disorder_sweep.npz", res=np.array(res, dtype=object))
    print("saved results/disorder_sweep.npz")
    print("\nREAD: KILL if both channels AND 2nd-ord err are flat/trivially-monotone in sigma (rel.range small,")
    print("endpoint peaks). SURVIVES only if non-monotonic intermediate-sigma structure tied to F_peak/t_peak.")
