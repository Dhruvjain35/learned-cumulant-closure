"""Shared container for baseline closures -- lives in its own module so pickles resolve from any script
(defining it inside a __main__ made the artifacts unloadable elsewhere; found the hard way)."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from closure_v3 import LocalClosureV3


class LocalClosureShell(LocalClosureV3):
    """Same container/applier contract as LocalClosureV3 (scalers/nets/range) with a custom regressor.
    The factory is passed to fit(), never stored -- instances must stay picklable."""
    def fit(self, data, make_net=None):
        self.types = sorted(data)
        for t, (X, Y) in data.items():
            if len(X) < 20:
                continue
            sc = StandardScaler().fit(X)
            net = make_net()
            net.fit(sc.transform(X), Y)
            self.scalers[t], self.nets[t] = sc, net
            self.range[t] = (Y.min(0), Y.max(0))


class FormulaNet:
    """A distilled closure formula: selected polynomial terms with least-squares weights. Picklable,
    sklearn-predict-compatible (consumed by the standard appliers after the production scaler)."""
    def __init__(self, poly, sel_idx, W, b, term_names):
        self.poly = poly                 # fitted PolynomialFeatures (transform only)
        self.sel_idx = np.asarray(sel_idx, int)
        self.W = np.asarray(W)           # (nsel, 2)
        self.b = np.asarray(b)           # (2,)
        self.term_names = list(term_names)

    def predict(self, Z):
        Phi = self.poly.transform(Z)[:, self.sel_idx]
        return Phi @ self.W + self.b
