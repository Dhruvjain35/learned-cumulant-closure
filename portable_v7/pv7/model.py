"""Frozen networks from models/{branch}/networks.npz, evaluated in NumPy.

The forward pass reproduces sklearn MLPRegressor.predict on a StandardScaler-transformed input operation for
operation (subtract mean, divide by scale, then per layer X @ W + b with ReLU between layers, identity output),
so results agree with the original estimator to floating-point rounding (bit-for-bit on the same machine)."""
import os
import json
import numpy as np
from . import PACKAGE_ROOT


class Net:
    __slots__ = ("W", "b", "mean", "scale", "lo", "hi")

    def __init__(self, W, b, mean, scale, lo, hi):
        self.W, self.b, self.mean, self.scale, self.lo, self.hi = W, b, mean, scale, lo, hi

    def predict_raw(self, X):
        """Unclipped network output for raw (ratio-transformed, unscaled) features X of shape (n, width)."""
        a = (np.asarray(X, dtype=np.float64) - self.mean) / self.scale
        last = len(self.W) - 1
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            a = a @ W
            a += b
            if i != last:
                np.maximum(a, 0, out=a)
        return a


class Branch:
    """All canonical-type networks of one frozen branch ('v4' or 'v6')."""

    def __init__(self, name, root=PACKAGE_ROOT):
        d = os.path.join(root, "models", name)
        with open(os.path.join(d, "metadata.json")) as fh:
            self.meta = json.load(fh)
        z = np.load(os.path.join(d, "networks.npz"), allow_pickle=False)
        self.width = self.meta["input_width"]
        self.nets = {}
        for t in self.meta["types"]:
            k = t["key"]
            nl = len(t["layers"])
            self.nets[tuple(t["type"])] = Net([z[f"{k}__W{i}"] for i in range(nl)],
                                              [z[f"{k}__b{i}"] for i in range(nl)],
                                              z[f"{k}__mean"], z[f"{k}__scale"], z[f"{k}__lo"], z[f"{k}__hi"])
