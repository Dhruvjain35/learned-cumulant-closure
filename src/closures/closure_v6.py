"""
V6: the pre-registered memory-aware closure (PROTOCOL.md "V6 PRE-REGISTRATION").

V6 = the V3-RATIO architecture + three physically-timed exponential memory traces
    h_x' = lambda_x (x - h_x)
for the context signals x = (nc/M, mean z, mean |coh|), with rates taken from the EOM, not tuned:
    lambda_nc = kappa;  lambda_coh = gdown/2 + gphi (transverse decay);  lambda_z = gdown.
Features appended per signal: (h_x, x - h_x) -> 27 total. Traces are initialized at their t=0 values
(no fake transients). In training, traces are computed along the exact moment series by the exact
constant-x exponential update per step (dt = 0.2; right-endpoint x; the in-loop trace is integrated
continuously by the ODE solver -- the small method mismatch is O(lambda*dt) ~ <1% and disclosed).
In the loop, traces are additional ODE state; the applier reads them via .current_h (set by the rhs).

Hermiticity: traces are real scalars entering features only; the conjugation structure (key-level
canonicalization, csign) is inherited unchanged from ClosureApplyV3.
"""
import numpy as np
from closure_v3 import ClosureApplyV3, context_v3, ratio_transform  # noqa: F401 (ratio re-exported)
from local_closure import extract

GDOWN = 0.02


def lambdas(kappa, gphi, gdown=GDOWN):
    """(lambda_nc, lambda_z, lambda_coh) -- order matches context_v3's (nc/M, mean z, mean coh)."""
    return np.array([kappa, gdown, gdown / 2.0 + gphi])


def trace_series(ctx_series, lam, dt):
    """Exponential update along a sampled signal with MIDPOINT x per step (second-order accurate;
    the earlier right-endpoint version carried an O(lambda*dt*xdot) lag bias of ~1-2% that check_v6
    caught as a systematic train-vs-loop mismatch). ctx_series: (NT, nsig). h0 = x0."""
    decay = np.exp(-lam * dt)
    h = np.empty_like(ctx_series)
    h[0] = ctx_series[0]
    for k in range(1, len(ctx_series)):
        xbar = 0.5 * (ctx_series[k - 1] + ctx_series[k])
        h[k] = xbar + (h[k - 1] - xbar) * decay
    return h


def trace_features(x, h):
    """[h_nc, x_nc - h_nc, h_z, x_z - h_z, h_coh, x_coh - h_coh]"""
    return [h[0], x[0] - h[0], h[1], x[1] - h[1], h[2], x[2] - h[2]]


def extract_v6(mdict, plan, det, ctx, regime, tr):
    return extract(mdict, plan, det, ctx) + list(regime) + list(tr)   # 21 + 6 = 27


class ClosureApplyV6(ClosureApplyV3):
    """Trace-augmented applier. The integrator's rhs sets .current_h (3 trace values) before each call."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.current_h = None

    def __call__(self, m):
        eng = self.eng
        mdict = {k: m[j] for j, k in enumerate(eng.track)}
        ctx = np.array(context_v3(mdict, eng.M))
        tr = np.array(trace_features(ctx, self.current_h))
        out = np.zeros(self.out_len, complex)
        for t, (oi, sing, pair, dfeat, csign) in self.blocks.items():
            if t not in self.lc.nets:
                continue
            s = m[sing]; p = m[pair]
            nk = s.shape[0]
            X = np.empty((nk, 27))
            X[:, 0:6:2] = s.real;  X[:, 1:6:2] = s.imag
            X[:, 6:12:2] = p.real; X[:, 7:12:2] = p.imag
            X[:, 12:15] = dfeat
            X[:, 15:18] = ctx
            X[:, 18:21] = self.regime
            X[:, 21:27] = tr
            if self.xform is not None:
                X = self.xform(X)                 # ratio transform touches cols 12:15, 18:21 only
            V = self.lc.nets[t].predict(self.lc.scalers[t].transform(X))
            lo, hi = self.lc.range[t]
            Vc = np.clip(V, lo, hi)
            if self.stats is not None:
                self.stats.setdefault("clip_n", 0); self.stats.setdefault("tot_n", 0)
                self.stats["clip_n"] += int(np.sum(Vc != V)); self.stats["tot_n"] += V.size
                if self.stats_B is not None:
                    r = np.abs(Vc) / self.stats_B[t]
                    self.stats.setdefault("r90", 0); self.stats.setdefault("r99", 0)
                    self.stats.setdefault("rN", 0); self.stats.setdefault("rmax", 0.0)
                    self.stats["r90"] += int(np.sum(r > 0.90)); self.stats["r99"] += int(np.sum(r > 0.99))
                    self.stats["rN"] += r.size
                    self.stats["rmax"] = max(self.stats["rmax"], float(np.max(r)))
            out[oi] = Vc[:, 0] + 1j * csign * Vc[:, 1]
        return out


def run_v6(eng, m0, clo, lam, tmax=40.0, nt=200, rtol=1e-4, atol=1e-6):
    """Closed loop with traces as additional ODE state y = [m.re, m.im, h(3)]; h(0) = ctx(m0)."""
    from scipy.integrate import solve_ivp
    ts = np.linspace(0, tmax, nt)
    nm = len(m0)
    h0 = np.array(context_v3({k: m0[i] for i, k in enumerate(eng.track)}, eng.M))

    def rhs(t, y):
        m = y[:nm] + 1j * y[nm:2 * nm]
        h = y[2 * nm:]
        clo.current_h = h
        kap = clo(m)
        dm = eng.deriv_compiled(m, kap)
        x = np.array(context_v3({k: m[i] for i, k in enumerate(eng.track)}, eng.M))
        return np.concatenate([dm.real, dm.imag, lam * (x - h)])

    sol = solve_ivp(rhs, (ts[0], ts[-1]), np.concatenate([m0.real, m0.imag, h0]),
                    t_eval=ts, rtol=rtol, atol=atol, method="RK45")
    traj = np.array([sol.y[:nm, i] + 1j * sol.y[nm:2 * nm, i] for i in range(sol.y.shape[1])])
    h_traj = sol.y[2 * nm:, :].T                     # the loop's OWN integrated traces (never discard:
    return ts, traj, h_traj                          # review found check(4) was blind without them)
