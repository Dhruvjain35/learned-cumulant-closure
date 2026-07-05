"""
Self-checks for the v3 closure machinery. Run BEFORE any training on v3 data.

(1) HERMITICITY OF TARGETS: on exact moments, kappa3(conj_key(k)) == conj(kappa3(k)) for every order-3 k.
    This is the identity that justifies the conjugate-canonical training filter; must hold to solver tol.
(2) APPLIER EQUIVALENCE: ClosureApplyV3's vectorized prediction equals a plain per-key reference
    implementation (extract_v3 + predict + conjugation) on the same moment vector, exactly.
(3) APPLIER HERMITICITY (the check the adversarial review proved (1)+(2) cannot provide): on an ARBITRARY
    moment vector, the applied closure satisfies out(conj k) == conj(out(k)) exactly for every conjugate
    pair present in eng.o3_keys -- including within self-conjugate signatures like (sp,sm,sz) -- and
    Im(out) == 0 exactly for self-conjugate keys. This is the structural guarantee, tested at the output.
"""
import numpy as np
from cumulant import mono_order, closure_terms
from scalable_engine import ScalableEngine, enum_sparse
from closure_v3 import (LocalClosureV3, ClosureApplyV3, conj_key, canonical_sig,
                        context_v3, extract_v3, add_samples_v3)
from gen_data_v3 import moments_v3, WC, GDOWN
from local_closure import feature_plan
from exact_tc import draw_disorder

GC, KA, GP, SG, SD = 0.7, 0.15, 0.02, 0.15, 0
M, NC = 3, 10

w = draw_disorder(M, WC, SG, SD, "gauss")
g = np.full(M, GC / np.sqrt(M))
keys, moms, nc_peak = moments_v3(M, NC, w, g, KA, GP, {})
o3 = [k for k in keys if mono_order(k) == 3]


def kap3(mdict, k):
    terms, _ = closure_terms(k)
    val = 0j
    for coeff, kk in terms:
        prod = coeff
        for f in kk:
            prod *= mdict[f]
        val += prod
    return mdict[k] - val


print("(1) hermiticity of kappa3 targets on exact moments:")
worst = 0.0
for ti in range(0, 200, 25):
    mdict = {k: moms[k][ti] for k in keys}
    for k in o3:
        worst = max(worst, abs(kap3(mdict, conj_key(k)) - np.conj(kap3(mdict, k))))
print(f"    max |kappa3(conj k) - conj(kappa3(k))| = {worst:.2e}  -> {'PASS' if worst < 1e-8 else 'FAIL'}")

print("(2) vectorized ClosureApplyV3 == per-key reference:")
# micro-train a v3 closure on this one config's samples (correctness check only, not a physics model)
Xd, Yd = {}, {}
add_samples_v3(M, keys, lambda ti: {k: moms[k][ti] for k in keys}, range(0, 200, 8),
               w - WC, (GC, KA, GP), Xd, Yd)
lc = LocalClosureV3()
lc.fit({t: (np.array(Xd[t]), np.array(Yd[t])) for t in Xd})

eng = ScalableEngine(4, 12, WC, draw_disorder(4, WC, SG, 7, "gauss"),
                     np.full(4, GC / 2.0), kappa=KA, gdown=GDOWN, gphi=GP, Fsafe=6).compile()
det4 = draw_disorder(4, WC, SG, 7, "gauss") - WC
regime = (GC, KA, GP)
app = ClosureApplyV3(lc, eng, det4, regime)

rng = np.random.default_rng(3)
m = 0.2 * (rng.standard_normal(len(eng.track)) + 1j * rng.standard_normal(len(eng.track)))
vec = app(m)

mdict = {k: m[i] for i, k in enumerate(eng.track)}
ctx = context_v3(mdict, eng.M)
ref = np.zeros(len(eng.o3_keys), complex)
for i, k in enumerate(eng.o3_keys):
    sig, conj = canonical_sig(k)
    if sig not in lc.nets:
        continue
    kk = conj_key(k) if conj else k
    X = np.array([extract_v3(mdict, feature_plan(kk), det4, ctx, regime)])
    V = lc.nets[sig].predict(lc.scalers[sig].transform(X))
    lo, hi = lc.range[sig]
    V = np.clip(V, lo, hi)
    if k == conj_key(k):
        ref[i] = V[0, 0]                       # self-conjugate: kappa3 exactly real (csign = 0)
    else:
        z = V[0, 0] + 1j * V[0, 1]
        ref[i] = np.conj(z) if conj else z
err = float(np.max(np.abs(vec - ref)))
print(f"    max |vectorized - reference| = {err:.2e}  -> {'PASS' if err < 1e-12 else 'FAIL'}")

print("(3) applier hermiticity on ARBITRARY input (structural, incl. within self-conjugate sigs):")
idx = {k: i for i, k in enumerate(eng.o3_keys)}
worst_pair, worst_self, n_pairs, n_self, n_absent = 0.0, 0.0, 0, 0, 0
for i, k in enumerate(eng.o3_keys):
    kc = conj_key(k)
    if kc == k:
        n_self += 1
        worst_self = max(worst_self, abs(vec[i].imag))
    elif kc in idx:
        n_pairs += 1
        worst_pair = max(worst_pair, abs(vec[idx[kc]] - np.conj(vec[i])))
    else:
        n_absent += 1
ok3 = worst_pair == 0.0 and worst_self == 0.0
print(f"    {n_pairs} paired keys: max |out(conj k) - conj(out(k))| = {worst_pair:.2e}")
print(f"    {n_self} self-conjugate keys: max |Im(out)| = {worst_self:.2e}   ({n_absent} partners absent)")
print(f"    -> {'PASS' if ok3 else 'FAIL'}")
