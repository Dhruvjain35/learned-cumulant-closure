"""Frozen V7 router, read from router.json."""
import os
import json
from . import PACKAGE_ROOT

with open(os.path.join(PACKAGE_ROOT, "router.json")) as _fh:
    ROUTER = json.load(_fh)
KAPPA_MAX, GAMMA_PHI_MAX = ROUTER["kappa_max"], ROUTER["gamma_phi_max"]


def route(kappa, gamma_phi):
    return "v6" if (kappa <= KAPPA_MAX and gamma_phi <= GAMMA_PHI_MAX) else "v4"
