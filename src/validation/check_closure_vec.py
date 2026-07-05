"""Confirm the vectorized ClosureApply produces identical kappa3 to the reference per-key extract() path."""
import numpy as np
from scalable_engine import ScalableEngine
from local_closure import LocalClosure, feature_plan, extract
from local_run import ClosureApply
from exact_tc import draw_disorder

WC, GCOLL, KAPPA, GDOWN, GPHI, NC = 1.0, 0.8, 0.1, 0.02, 0.02, 12

lc = LocalClosure(M=3)
lc.fit(np.load("results/transfer_train.npz", allow_pickle=True)["data"].item())

M = 4
w = draw_disorder(M, WC, 0.15, 7, "gauss")
g = np.full(M, GCOLL / np.sqrt(M))
eng = ScalableEngine(M, NC, WC, w, g, kappa=KAPPA, gdown=GDOWN, gphi=GPHI, Fsafe=6).compile()
det = w - WC

rng = np.random.default_rng(0)
m = 0.2 * (rng.standard_normal(len(eng.track)) + 1j * rng.standard_normal(len(eng.track)))

# vectorized
kap_vec = ClosureApply(lc, eng, det)(m)

# reference: per-key extract + predict (same math as the old loop)
mdict = {k: m[i] for i, k in enumerate(eng.track)}
ctx = lc.context(mdict, eng.M)
ref = np.zeros(len(eng.o3_keys), complex)
plans = [feature_plan(k) for k in eng.o3_keys]
by_type = {}
for i, k in enumerate(eng.o3_keys):
    by_type.setdefault(plans[i][0], []).append(i)
for t, idxs in by_type.items():
    if t not in lc.nets:
        continue
    X = np.array([extract(mdict, plans[i], det, ctx) for i in idxs])
    V = lc.nets[t].predict(lc.scalers[t].transform(X))
    lo, hi = lc.range[t]
    V = np.clip(V, lo, hi)
    for r, i in enumerate(idxs):
        ref[i] = V[r, 0] + 1j * V[r, 1]

err = float(np.max(np.abs(kap_vec - ref)))
print(f"max|vectorized - per-key extract| kappa3 = {err:.2e}  ->  {'PASS' if err < 1e-12 else 'FAIL'}")
