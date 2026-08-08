"""
GATE 1 for the MPS baseline: does the site-decomposed Liouvillian equal QuTiP's?

The MPS solver never builds the Liouvillian densely -- it builds it as a sum of onsite terms and
two-site cavity-emitter terms, which is what lets it become an MPO. This script assembles those
same pieces densely at small M and checks them against qutip.liouvillian() for the identical
model. If this fails, nothing downstream is trustworthy.

One subtlety is handled here explicitly. Per-site doubling orders the vectorised index as
(col,row) nested per site: (c_cav,r_cav, c_1,r_1, ...). Global vectorisation orders it as
(all cols, then all rows): (c_cav,c_1,...,r_cav,r_1,...). The two are related by a transpose of
tensor axes, applied in `site_to_global` below. Getting this wrong produces a matrix with the
right spectrum and the wrong dynamics, so it is checked rather than assumed.

Run:  python src/validation/check_mps_liouvillian.py
"""
import sys
import os

import numpy as np
import qutip as qt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "baselines"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "engines"))

from mps_lindblad import cavity_ops, emitter_ops, op_L, op_R, dissipator_super  # noqa: E402


def site_to_global(Mat, dims):
    """Reorder a per-site-doubled operator into global column-stacking order.

    dims lists the UNDOUBLED physical dimension of each site, cavity first.
    """
    n = len(dims)
    # axes of the per-site index, in order: (c_0, r_0, c_1, r_1, ...)
    shape = []
    for d in dims:
        shape += [d, d]
    full = shape + shape                       # row-index axes then col-index axes of the matrix
    T = Mat.reshape(full)
    # within each half, gather all c axes then all r axes
    c_axes = [2 * i for i in range(n)]
    r_axes = [2 * i + 1 for i in range(n)]
    perm = c_axes + r_axes
    perm = perm + [p + 2 * n for p in perm]
    T = T.transpose(perm)
    D = int(np.prod(dims))
    return T.reshape(D * D, D * D)


def dense_from_sites(M, Nc, wc, w, g, kappa, gdown, gphi):
    """Assemble the Liouvillian from exactly the pieces the MPO uses."""
    cav, emi = cavity_ops(Nc), emitter_ops()
    dims = [Nc] + [2] * M
    dd = [d * d for d in dims]

    def embed(op, site):
        mats = [np.eye(x, dtype=complex) for x in dd]
        mats[site] = op
        out = mats[0]
        for m in mats[1:]:
            out = np.kron(out, m)
        return out

    def embed2(opA, sa, opB, sb):
        mats = [np.eye(x, dtype=complex) for x in dd]
        mats[sa], mats[sb] = opA, opB
        out = mats[0]
        for m in mats[1:]:
            out = np.kron(out, m)
        return out

    L = np.zeros((int(np.prod(dd)),) * 2, dtype=complex)

    Lc = -1j * (wc * op_L(cav["n"], Nc) - wc * op_R(cav["n"], Nc))
    if kappa > 0:
        Lc = Lc + dissipator_super(np.sqrt(kappa) * cav["a"], Nc)
    L += embed(Lc, 0)

    for i in range(M):
        Li = -1j * (w[i] * op_L(emi["ee"], 2) - w[i] * op_R(emi["ee"], 2))
        if gdown > 0:
            Li = Li + dissipator_super(np.sqrt(gdown) * emi["sm"], 2)
        if gphi > 0:
            Li = Li + dissipator_super(np.sqrt(gphi / 2.0) * emi["sz"], 2)
        L += embed(Li, i + 1)

    for i in range(M):
        for cop, eop in (("a", "sp"), ("ad", "sm")):
            L += embed2(-1j * g[i] * op_L(cav[cop], Nc), 0, op_L(emi[eop], 2), i + 1)
            L += embed2(+1j * g[i] * op_R(cav[cop], Nc), 0, op_R(emi[eop], 2), i + 1)

    return site_to_global(L, dims)


def qutip_liouvillian(M, Nc, wc, w, g, kappa, gdown, gphi):
    """The same model straight out of the repo's exact solver."""
    from exact_tc import build_ops, build_H, collapse_ops
    ops = build_ops(M, Nc)
    H = build_H(ops, wc, w, g)
    c = collapse_ops(ops, kappa=kappa, gphi=gphi, gdown=gdown)
    return qt.liouvillian(H, c).full()


def main():
    wc, gdown = 1.0, 0.02
    rng = np.random.default_rng(3)
    ok = True
    for M, Nc in ((1, 4), (2, 3), (3, 3)):
        w = wc + rng.normal(0.0, 0.15, M)
        gc, kappa, gphi = 0.7, 0.03, 0.02
        g = np.full(M, gc / np.sqrt(M))
        A = dense_from_sites(M, Nc, wc, w, g, kappa, gdown, gphi)
        B = qutip_liouvillian(M, Nc, wc, w, g, kappa, gdown, gphi)
        diff = np.abs(A - B).max()
        scale = np.abs(B).max()
        good = diff < 1e-12 * max(scale, 1.0)
        ok &= good
        print(f"  M={M} Nc={Nc}: max|L_sites - L_qutip| = {diff:.3e}   "
              f"(scale {scale:.3f})  {'PASS' if good else 'FAIL'}")
    print("\nGATE 1", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
