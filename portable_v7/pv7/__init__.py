"""
pv7 -- portable inference for the FROZEN V7 learned third-cumulant closure.

No sklearn, no pickle: the learned networks are read from models/{v4,v6}/networks.npz (np.load with
allow_pickle=False) and evaluated in NumPy. The moment engine in pv7/engine/ is vendored verbatim from the source
repository (only its flat imports were made package-relative). Every function here mirrors a named function in the
source repository; tests/test_portable_v7.py (in the source repository) checks them against the originals.
"""
import os

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
