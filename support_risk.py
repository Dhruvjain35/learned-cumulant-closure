"""
Support-risk indicator + fail-safe hybrid closure (roadmap item: "the simulator knows when its learned
physics is outside support and falls back safely").

RISK SCORE (per closure call, per dominant type): Mahalanobis-style distance of the standardized
(ratio-transformed) feature rows from the training distribution, using a shrunk empirical covariance
fitted ONCE on training features. Reported as d = sqrt(mahalanobis^2 / n_features), so d ~ 1 means
"typical training-like state" and d >> 1 means "outside learned support". This is a SUPPORT-RISK
INDICATOR, not calibrated uncertainty, and is named accordingly everywhere.

FAIL-SAFE HYBRID: kappa3_used = (1 - w(d)) * kappa3_NN, with w(d) = sigmoid((d - d0)/s) and d0 set to the
99th percentile of the training data's own self-distance (fixed a priori; s = 0.25*d0). As w -> 1 the
learned correction fades to zero, i.e. the rollout falls back to plain second-order cumulant behavior
instead of extrapolating the network.

Validation plan (separate run, dev seeds 7/8 first, untouched seeds for the verdict): does risk at time t
predict blowup at t + dt? Does the hybrid keep in-support accuracy while surviving the pocket?
"""
import numpy as np
from closure_v3 import ClosureApplyV3, ratio_transform


class SupportModel:
    def __init__(self, lc, data_ratio, max_rows=50_000, shrink=0.05, seed=0):
        """lc: the trained LocalClosureV3 (its scalers define the standardized space).
        data_ratio: the ratio-transformed training dict the lc was fitted on."""
        rng = np.random.default_rng(seed)
        self.icov, self.mu, self.d0 = {}, {}, {}
        for t, sc in lc.scalers.items():
            X = data_ratio[t][0]
            if len(X) > max_rows:
                X = X[rng.choice(len(X), max_rows, replace=False)]
            Z = sc.transform(X)
            mu = Z.mean(0)
            C = np.cov(Z.T) + shrink * np.eye(Z.shape[1])
            icov = np.linalg.inv(C)
            self.mu[t], self.icov[t] = mu, icov
            dz = Z - mu
            d = np.sqrt(np.einsum("ij,jk,ik->i", dz, icov, dz) / Z.shape[1])
            self.d0[t] = float(np.percentile(d, 99))            # in-support boundary, fixed a priori

    def score(self, t, Z):
        dz = Z - self.mu[t]
        return np.sqrt(np.einsum("ij,jk,ik->i", dz, self.icov[t], dz) / Z.shape[1])


class SafeApply(ClosureApplyV3):
    """Hybrid closure: learned kappa3 faded toward zero (= plain 2nd-order cumulant) as the local state
    leaves training support. Fade weight w = sigmoid((d - d0)/(0.25 d0)) per feature row. Records
    (t, max d/d0) when .risk_log is a list and .current_t is set by the integrator wrapper."""
    def __init__(self, *a, support=None, fade=True, **kw):
        """fade=False: record risk but do NOT alter the closure output -- used to VALIDATE the indicator
        along the base model's own trajectory (does risk rise before the base model's blowup?)."""
        super().__init__(*a, **kw)
        self.support = support
        self.fade = fade
        self.risk_log = None
        self.current_t = None

    def __call__(self, m):
        # replicate parent flow but scale V by (1 - w) before bounding; simplest correct route:
        # compute parent output per type via parent __call__ machinery is monolithic, so we instead
        # post-scale: get parent kappa3, then recompute per-type fade from the SAME feature rows.
        eng = self.eng
        from closure_v3 import context_v3
        mdict = {k: m[j] for j, k in enumerate(eng.track)}
        ctx = np.array(context_v3(mdict, eng.M))
        out = np.zeros(self.out_len, complex)
        worst = 0.0
        for t, (oi, sing, pair, dfeat, csign) in self.blocks.items():
            if t not in self.lc.nets:
                continue
            s = m[sing]; p = m[pair]
            nk = s.shape[0]
            X = np.empty((nk, 21))
            X[:, 0:6:2] = s.real;  X[:, 1:6:2] = s.imag
            X[:, 6:12:2] = p.real; X[:, 7:12:2] = p.imag
            X[:, 12:15] = dfeat
            X[:, 15:18] = ctx
            X[:, 18:21] = self.regime
            if self.xform is not None:
                X = self.xform(X)
            Z = self.lc.scalers[t].transform(X)
            V = self.lc.nets[t].predict(Z)
            if self.bound_mode == "tanh":
                B = self.rbound[t]
                V = B * np.tanh(V / B)
            else:
                lo, hi = self.lc.range[t]
                V = np.clip(V, lo, hi)
            if self.support is not None:
                d = self.support.score(t, Z)
                d0 = self.support.d0[t]
                if self.fade:
                    w = 1.0 / (1.0 + np.exp(-(d - d0) / (0.25 * d0)))
                    V = V * (1.0 - w)[:, None]
                worst = max(worst, float(np.max(d / d0)))
            out[oi] = V[:, 0] + 1j * csign * V[:, 1]
        if self.risk_log is not None:
            self.risk_log.append((self.current_t, worst))
        return out
