"""Feature construction, mirroring src/closures/local_closure.py (factor_info, canonical, feature_plan) and
src/closures/closure_v3.py (conj_key, canonical_sig, context_v3, ratio_transform) and closure_v6.py (lambdas,
trace_features). Logic copied line for line; only the sklearn-dependent imports are gone."""
import numpy as np
from .engine.cumulant import factors_of, pair_key

TYPE_ORDER = {"ad": 0, "a": 1, "sp": 2, "sm": 3, "sz": 4}
CONJE = {0: 0, 1: 2, 2: 1, 3: 3}
GDOWN = 0.02


def factor_info(fk):
    p, q, es = fk
    if p == 1 and q == 0:
        return "ad", -1
    if p == 0 and q == 1:
        return "a", -1
    for i, e in enumerate(es):
        if e != 0:
            return {1: "sp", 2: "sm", 3: "sz"}[e], i
    raise ValueError(fk)


def canonical(mono_key):
    facs = factors_of(mono_key)
    info = [factor_info(f) for f in facs]
    order = sorted(range(3), key=lambda k: (TYPE_ORDER[info[k][0]], info[k][1]))
    fk = [facs[k] for k in order]
    lab = [info[k] for k in order]
    sig = tuple(l[0] for l in lab)
    return sig, fk, lab


def feature_plan(mono_key):
    sig, fk, lab = canonical(mono_key)
    singles = fk
    pairs = [pair_key(fk[0], fk[1]), pair_key(fk[0], fk[2]), pair_key(fk[1], fk[2])]
    sites = [l[1] for l in lab]
    return sig, singles, pairs, sites


def conj_key(k):
    p, q, es = k
    return (q, p, tuple(CONJE[e] for e in es))


def canonical_sig(k):
    s = feature_plan(k)[0]
    kc = conj_key(k)
    sc = feature_plan(kc)[0]
    if s < sc:
        return (s, False)
    if s > sc:
        return (sc, True)
    return (s, False) if k <= kc else (s, True)


def context_v3(mdict, M):
    nc = mdict[(1, 1, tuple([0] * M))].real / M
    zs = [mdict[(0, 0, tuple(3 if s == i else 0 for s in range(M)))].real for i in range(M)]
    coh = [abs(mdict[(0, 0, tuple({i: 1, j: 2}.get(s, 0) for s in range(M)))])
           for i in range(M) for j in range(M) if i != j]
    return [nc, float(np.mean(zs)), float(np.mean(coh)) if coh else 0.0]


def ratio_transform(X):
    X = np.array(X, float, copy=True)
    ka = X[:, 19:20]
    X[:, 12:15] /= ka
    X[:, 18:19] /= ka
    X[:, 20:21] /= ka
    return X


def lambdas(kappa, gphi, gdown=GDOWN):
    return np.array([kappa, gdown, gdown / 2.0 + gphi])


def trace_features(x, h):
    return [h[0], x[0] - h[0], h[1], x[1] - h[1], h[2], x[2] - h[2]]
