"""
Does retraining the closure on larger M (now reachable thanks to the scalable engine) recover large-M
accuracy? Head-to-head, everything else held fixed:

  - lc_old: trained on M=2,3 only        (results/transfer_train.npz  -- the Result 6/8 closure)
  - lc_new: trained on M=2,3,4,5         (results/transfer_train_v2.npz from gen_data_v2.py)

Evaluated in the closed loop with the SAME validated scalable engine, SAME held-out disorder (seed 7,
sigma 0.15, never in training), SAME seeded mcsolve references as Result 8 (identical seeds/ntraj, so the
lc_old rows must REPRODUCE Result 8's numbers to the digit -- a built-in regression check).

Metrics: abs err = max over t of |obs - ref| for n and mean-sz (comparable to all previous tables), plus
rel_n = max|n - n_ref| / n_peak_ref to separate accuracy from the M-growing superradiant scale.
"""
import os
import time
import numpy as np
from local_closure import LocalClosure
from local_run import run, ClosureApply, observables, stable_ok, exact_dense_obs, build
from mc_ref import mc_reference

WC, GCOLL, KAPPA, GDOWN, GPHI = 1.0, 0.8, 0.1, 0.02, 0.02
NC = 12
TEST_SEED, TEST_SIGMA = 7, 0.15
NTRAJ = 200                                     # same as Result 8's large_ref


def fit_closure(cache):
    lc = LocalClosure(M=3)
    lc.fit(np.load(cache, allow_pickle=True)["data"].item())
    dom = {t: round(lc.nets[t].best_validation_score_, 3) for t in lc.types
           if t in lc.nets and lc.nets[t].best_validation_score_ > 0.8}
    return lc, dom


def errs(eng, traj, nref, zref):
    if not stable_ok(traj):
        return float("inf"), float("inf")
    n, z = observables(eng, traj)
    L = min(len(n), len(nref))
    ea = max(float(np.max(np.abs(n[:L] - nref[:L]))), float(np.max(np.abs(z[:L] - zref[:L]))))
    rn = float(np.max(np.abs(n[:L] - nref[:L]))) / float(np.max(nref))
    return ea, rn


if __name__ == "__main__":
    # fail fast BEFORE the ~10-min lc_old fit: both caches must exist (v2 comes from gen_data_v2.py)
    for cache in ("results/transfer_train.npz", "results/transfer_train_v2.npz"):
        assert os.path.exists(cache), f"missing {cache} -- run gen_data_v2.py first"
    lc_old, dom_old = fit_closure("results/transfer_train.npz")
    print(f"lc_old (M=2,3):      dominant types {dom_old}")
    lc_new, dom_new = fit_closure("results/transfer_train_v2.npz")
    print(f"lc_new (M=2,3,4,5):  dominant types {dom_new}\n")

    print("closed loop, held-out disorder (seed 7, sigma 0.15), vs exact reference:")
    print(f"{'M':>3s} {'ref':>8s} {'plain':>8s} {'old(2,3)':>9s} {'new(2-5)':>9s} "
          f"{'rel_n old':>10s} {'rel_n new':>10s} {'MCerr':>6s} {'sec':>5s}")
    rows = []
    for M, ref_kind in [(5, "mesolve"), (8, "mcsolve"), (10, "mcsolve")]:
        t0 = time.time()
        eng, w, g = build(M, TEST_SIGMA, TEST_SEED)
        det = w - WC
        _, tp = run(eng, None)
        _, to = run(eng, ClosureApply(lc_old, eng, det))
        _, tn = run(eng, ClosureApply(lc_new, eng, det))
        if ref_kind == "mesolve":
            nref, zref = exact_dense_obs(M, NC, w, g)
            mcerr = 0.0
        else:
            _, nref, zref, mcerr = mc_reference(M, NC, w, g, ntraj=NTRAJ)   # same seeds/ntraj as Result 8
        ep, _ = errs(eng, tp, nref, zref)
        eo, ro = errs(eng, to, nref, zref)
        en, rn = errs(eng, tn, nref, zref)
        note = "(M=5 IS in new training; held-out seed only)" if M == 5 else ""
        print(f"{M:>3d} {ref_kind:>8s} {ep:>8.4f} {eo:>9.4f} {en:>9.4f} {ro:>10.3f} {rn:>10.3f} "
              f"{mcerr:>6.3f} {time.time()-t0:>5.0f}  {note}")
        rows.append((M, ep, eo, en, ro, rn, mcerr))

    print("\nstability of lc_new at very large M (no exact reference possible):")
    print("   NOTE: these sizes are outside the validated operating range (see")
    print("   src/closures/validated_range.py). z_end below is a MEAN over emitters and")
    print("   cannot detect the failure mode measured for the frozen V7 at M>=12, where")
    print("   per-emitter inversions diverge with opposite signs and leave the mean near -1.")
    print("   Per-emitter bounds are therefore reported alongside it.")
    from validated_range import check, status
    for M in [12, 16, 20]:
        t0 = time.time()
        check(M, mode="warn")
        eng, w, g = build(M, TEST_SIGMA, TEST_SEED)
        _, tn = run(eng, ClosureApply(lc_new, eng, w - WC))
        n, z = observables(eng, tn)
        # per-emitter inversion bound: the diagnostic the mean cannot provide
        zcols = [eng.tpos[(0, 0, tuple(3 if s == i else 0 for s in range(M)))] for i in range(M)]
        Z = np.array([[row[c].real for c in zcols] for row in tn])
        zviol = float(max(0.0, np.max(np.abs(Z)) - 1.0))
        physical = (zviol <= 1e-9) and (n.min() >= -1e-9)
        print(f"   M={M} [{status(M)}]: stable={stable_ok(tn)}  physical={physical}  "
              f"n_peak={n.max():.3f}  n_min={n.min():.3e}  z_end={z[-1]:.3f}  "
              f"max|z_i|-1={zviol:.3e}  ({time.time()-t0:.0f}s)")

    np.savez_compressed("results/retrain_eval.npz", rows=np.array(rows, dtype=object))
    print("\nsaved results/retrain_eval.npz")
