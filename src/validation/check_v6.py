"""
V6 self-checks. Run BEFORE the data regeneration and BEFORE any V6 training.

(1) TRACE CORRECTNESS: the discrete exponential update (trace_series) must match a 100x-finer reference
    integration of h' = lambda (x - h) with linearly interpolated x, within the documented O(lambda*dt)
    method tolerance (0.005), for the largest physical lambda in the grid (0.4).
(2) APPLIER EQUIVALENCE: ClosureApplyV6 vectorized == per-key reference (extract_v6 + predict + conj).
(3) APPLIER HERMITICITY on arbitrary input WITH traces: out(conj k) == conj(out(k)) exactly; Im == 0 for
    self-conjugate keys. (Traces are real feature inputs; the guarantee must survive them untouched.)
(4) IN-LOOP TRACE WIRING: run_v6 with a null closure (kappa3 = 0, i.e. plain cumulant dynamics) must
    produce integrated traces h(t) matching trace_series applied to the rollout's own ctx series, within
    the method tolerance -- validates the augmented ODE state plumbing end to end.
"""
import numpy as np
from scipy.integrate import solve_ivp

from closure_v3 import LocalClosureV3, context_v3, conj_key, canonical_sig, ratio_transform
from closure_v6 import (lambdas, trace_series, trace_features, extract_v6, ClosureApplyV6, run_v6)
from gen_data_v6 import add_samples_v6, DT
from gen_data_v3 import moments_v3, WC
from scalable_engine import ScalableEngine
from local_closure import feature_plan
from pilot_eval import m0_product
from exact_tc import draw_disorder

print("(1) trace update vs 100x-finer reference (lambda = 0.4, worst case):")
ts = np.arange(0, 40 + 1e-9, DT)
x = np.sin(0.7 * ts) * 0.5 + 0.5
lam = 0.4
h_disc = trace_series(x[:, None], np.array([lam]), DT)[:, 0]
tf = np.arange(0, 40 + 1e-9, DT / 100)
xf = np.interp(tf, ts, x)
hf = trace_series(xf[:, None], np.array([lam]), DT / 100)[:, 0]
err1 = float(np.max(np.abs(h_disc - hf[::100])))
print(f"    max diff = {err1:.2e}  -> {'PASS' if err1 < 5e-3 else 'FAIL'}")

print("(2)+(3) applier equivalence and hermiticity (micro-trained V6 nets, M=4 engine):")
GC, KA, GP, SG = 0.7, 0.15, 0.02, 0.15
w3 = draw_disorder(3, WC, SG, 0, "gauss")
keys3, moms3, _ = moments_v3(3, 10, w3, np.full(3, GC / np.sqrt(3)), KA, GP, {})
Xd, Yd, Pd = {}, {}, {}
add_samples_v6(3, keys3, moms3, w3 - WC, (GC, KA, GP), lambdas(KA, GP), 0, Xd, Yd, Pd)
lc = LocalClosureV3()
lc.fit({t: (np.array(Xd[t]), np.array(Yd[t])) for t in Xd})

w4 = draw_disorder(4, WC, SG, 7, "gauss")
eng = ScalableEngine(4, 12, WC, w4, np.full(4, GC / 2.0), kappa=KA, gdown=0.02, gphi=GP, Fsafe=6).compile()
det = w4 - WC
app = ClosureApplyV6(lc, eng, det, (GC, KA, GP), xform=ratio_transform)
rng = np.random.default_rng(3)
m = 0.2 * (rng.standard_normal(len(eng.track)) + 1j * rng.standard_normal(len(eng.track)))
app.current_h = np.array([0.1, 0.4, 0.05])
vec = app(m)

mdict = {k: m[i] for i, k in enumerate(eng.track)}
ctx = context_v3(mdict, eng.M)
tr = trace_features(np.array(ctx), app.current_h)
ref = np.zeros(len(eng.o3_keys), complex)
for i, k in enumerate(eng.o3_keys):
    sig, conj = canonical_sig(k)
    if sig not in lc.nets:
        continue
    kk = conj_key(k) if conj else k
    X = ratio_transform(np.array([extract_v6(mdict, feature_plan(kk), det, ctx, (GC, KA, GP), tr)]))
    V = lc.nets[sig].predict(lc.scalers[sig].transform(X))
    lo, hi = lc.range[sig]
    V = np.clip(V, lo, hi)
    if k == conj_key(k):
        ref[i] = V[0, 0]
    else:
        z = V[0, 0] + 1j * V[0, 1]
        ref[i] = np.conj(z) if conj else z
err2 = float(np.max(np.abs(vec - ref)))
print(f"    vectorized vs reference: {err2:.2e}  -> {'PASS' if err2 < 1e-12 else 'FAIL'}")
idx = {k: i for i, k in enumerate(eng.o3_keys)}
worst_p, worst_s = 0.0, 0.0
for i, k in enumerate(eng.o3_keys):
    kc = conj_key(k)
    if kc == k:
        worst_s = max(worst_s, abs(vec[i].imag))
    elif kc in idx:
        worst_p = max(worst_p, abs(vec[idx[kc]] - np.conj(vec[i])))
ok3 = worst_p == 0.0 and worst_s == 0.0
print(f"    hermiticity with traces: pairs {worst_p:.2e}, self-conj imag {worst_s:.2e}  "
      f"-> {'PASS' if ok3 else 'FAIL'}")

print("(4) in-loop trace wiring: the LOOP'S OWN integrated h (returned by run_v6) vs references:")
# (review blocker fix: the old check compared trace_series vs an ODE of the same ctx -- it never observed
#  run_v6's internal h and passed under an injected lambda transposition. Now h_traj IS the loop state.)
class SpyClo:
    def __init__(self):
        self.first_h = None
        self.calls = 0
        self.current_h = None
    def __call__(self, m):
        assert self.current_h is not None, "current_h not set before closure call"
        if self.first_h is None:
            self.first_h = np.array(self.current_h)
        self.calls += 1
        return None
lam4 = lambdas(KA, GP)
m0 = m0_product(eng, 1.0)
spy = SpyClo()
ts4, traj, h_loop = run_v6(eng, m0, spy, lam4)
ctxs = np.array([context_v3({k: row[i] for i, k in enumerate(eng.track)}, eng.M) for row in traj])
h0_expect = context_v3({k: m0[i] for i, k in enumerate(eng.track)}, eng.M)
ok_h0 = np.allclose(spy.first_h, h0_expect, atol=1e-9) and np.allclose(h_loop[0], h0_expect, atol=1e-9)
print(f"    h(0) == ctx(m0) at first closure call and in h_traj  -> {'PASS' if ok_h0 else 'FAIL'}")
err4 = float(np.max(np.abs(h_loop - trace_series(ctxs, lam4, DT))))
print(f"    max |LOOP h - discrete trace of rollout ctx| = {err4:.2e}  "
      f"-> {'PASS' if err4 < 5e-3 else 'FAIL'}")
# per-channel independent references (any lambda misrouting inside rhs fails a named channel here)
for j, namej in enumerate(["nc", "z", "coh"]):
    ref = solve_ivp(lambda t, h: lam4[j] * (np.interp(t, ts4, ctxs[:, j]) - h),
                    (0, ts4[-1]), [ctxs[0, j]], t_eval=ts4, rtol=1e-9, atol=1e-11).y[0]
    ej = float(np.max(np.abs(h_loop[:, j] - ref)))
    print(f"    channel {namej}: max |LOOP h - independent ODE| = {ej:.2e}  "
          f"-> {'PASS' if ej < 5e-3 else 'FAIL'}")

print("(5) lambda literal binding (executable transcription of the pre-registration):")
# nc<->kappa = 0.17;  z<->gdown = 0.02;  coh<->gdown/2+gphi = 0.02/2 + 0.06 = 0.07
# (first run of this check FAILED on a hand-transcription error in the expected constant -- 0.05 instead
#  of 0.07; the implementation was correct, independently confirmed by sentinel part C. Kept as a record.)
ok5 = np.allclose(lambdas(0.17, 0.06), [0.17, 0.02, 0.07])
print(f"    lambdas(kappa=0.17, gphi=0.06) == [0.17, 0.02, 0.07]  -> {'PASS' if ok5 else 'FAIL'}")
