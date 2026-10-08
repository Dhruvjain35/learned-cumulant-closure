"""Portable in-loop closure appliers, mirroring ClosureApplyV3 / ClosureApplyV6 (src/closures/closure_v3.py,
closure_v6.py) in the default 'clip' mode used by V7. Same canonical routing, same feature rows, same clipping,
same conjugation; the only difference is that the network is evaluated by pv7.model instead of sklearn."""
import numpy as np
from .features import canonical_sig, conj_key, feature_plan, context_v3, ratio_transform, trace_features


class ClosureApply:
    def __init__(self, branch, eng, det, regime, memory=False):
        self.branch, self.eng = branch, eng
        self.regime = np.asarray(regime, float)
        self.memory = memory
        self.current_h = None
        tp = eng.tpos
        by_type = {}
        for i, k in enumerate(eng.o3_keys):
            sig, conj = canonical_sig(k)
            kk = conj_key(k) if conj else k
            _, singles, pairs, sites = feature_plan(kk)
            csign = 0.0 if k == conj_key(k) else (-1.0 if conj else 1.0)
            by_type.setdefault(sig, []).append(
                (i, [tp[s] for s in singles], [tp[p] for p in pairs],
                 [det[s] if s >= 0 else 0.0 for s in sites], csign))
        self.blocks = {}
        for t, rows in by_type.items():
            self.blocks[t] = (np.array([r[0] for r in rows]),
                              np.array([r[1] for r in rows]),
                              np.array([r[2] for r in rows]),
                              np.array([r[3] for r in rows]),
                              np.array([r[4] for r in rows]))
        self.out_len = len(eng.o3_keys)

    def features(self, m):
        """Ratio-transformed feature matrix per canonical type: {type: (output_indices, X, csign)}."""
        eng = self.eng
        mdict = {k: m[j] for j, k in enumerate(eng.track)}
        ctx = np.array(context_v3(mdict, eng.M))
        width = 27 if self.memory else 21
        tr = np.array(trace_features(ctx, self.current_h)) if self.memory else None
        feats = {}
        for t, (oi, sing, pair, dfeat, csign) in self.blocks.items():
            if t not in self.branch.nets:
                continue
            s = m[sing]; p = m[pair]
            nk = s.shape[0]
            X = np.empty((nk, width))
            X[:, 0:6:2] = s.real;  X[:, 1:6:2] = s.imag
            X[:, 6:12:2] = p.real; X[:, 7:12:2] = p.imag
            X[:, 12:15] = dfeat
            X[:, 15:18] = ctx
            X[:, 18:21] = self.regime
            if self.memory:
                X[:, 21:27] = tr
            feats[t] = (oi, ratio_transform(X), csign)
        return feats

    def __call__(self, m):
        out = np.zeros(self.out_len, complex)
        for t, (oi, X, csign) in self.features(m).items():
            net = self.branch.nets[t]
            V = net.predict_raw(X)
            Vc = np.clip(V, net.lo, net.hi)
            out[oi] = Vc[:, 0] + 1j * csign * Vc[:, 1]
        return out
