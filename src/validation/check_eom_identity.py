"""
EOM identity gate: prove the second-order equations of motion are derived correctly, in the cells
that actually carry the headline claim.

The check is an identity, not a tolerance-tuned comparison. For every tracked moment O:

    d<O>/dt  =  <Ldag[O]>            (exact, from the Lindblad generator)

and the closure engine assembles the same derivative from (a) the order<=2 moments and (b) the
third cumulants. Feed it the EXACT third cumulants and the approximation disappears, so the two
must agree to machine precision. If the projection coefficients, the moment-to-cumulant assembly,
or the operator algebra were wrong anywhere, this identity would break.

`cumulant.py` has carried a version of this since early on, run on one ad-hoc cell. This gate runs
it on the cells the paper reports (POCKET and CONTROL, M = 2 and 3, the disorder draws exported in
results/replication_params.json), at times spanning the superradiant burst, and writes a log.

WHAT THIS PROVES: given the Hamiltonian and jump operators as coded, the second-order EOMs and the
cumulant assembly are exact. Combined with check_conventions.py, which tests those operators
against closed-form results, the derivation chain is machine-checked end to end.

WHAT IT DOES NOT PROVE: that the coded model is the one a reader means by "open disordered
Tavis-Cummings." That is a human read; see docs/EXTERNAL_VALIDATION_STATUS.md.
"""
import json
import os

import numpy as np
import qutip as qt

from cumulant import CumulantEngine

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PARAMS = os.path.join(REPO, "results", "replication_params.json")
LOG = os.path.join(REPO, "results", "eom_identity.log")

NC = 12          # Fock cutoff for the exact state used to probe the identity
FSAFE = 6
TOL = 1e-8
TIMES = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0]     # spans the burst (t_peak ~ 2.6 to 3.0)


def run_cell(cell, out):
    M = cell["M"]
    w = np.array(cell["w"], float)
    g = np.array(cell["g"], float)
    eng = CumulantEngine(M, NC, cell["wc"], w, g,
                         kappa=cell["kappa"], gdown=cell["gdown"], gphi=cell["gphi"], Fsafe=FSAFE)
    psi0 = qt.tensor(qt.basis(NC, 0), *[qt.basis(2, 0)] * M)     # the burst IC: vacuum, all excited
    res = qt.mesolve(eng.H, psi0, TIMES, c_ops=eng.cops,
                     options={"progress_bar": False, "atol": 1e-12, "rtol": 1e-10})
    head = (f"{cell['regime']:<8} M={M}  seed={cell['seed']}  "
            f"kappa={cell['kappa']}  gphi={cell['gphi']}  "
            f"tracked={len(eng.track)}  order-3={len(eng.order3)}")
    out.append(head)
    print("   " + head)
    ok = True
    for t, rho in zip(TIMES, res.states):
        err, _, _ = eng.gate_exact(rho)
        p = err < TOL
        ok = ok and p
        line = f"      t={t:>4.1f}   max|d_assembled - d_exact| = {err:.3e}   {'PASS' if p else 'FAIL'}"
        out.append(line)
        print(line)
    return ok


def main():
    with open(PARAMS) as f:
        cells = json.load(f)["cells"]
    cells = [c for c in cells if c["M"] in (2, 3)]

    out = []
    out.append("EOM IDENTITY GATE  --  d<O>/dt = <Ldag[O]> for every tracked moment,")
    out.append("with EXACT third cumulants fed to the closure assembly (so no approximation remains).")
    out.append(f"Fock cutoff Nc={NC}, Fsafe={FSAFE}, tolerance {TOL:.0e}. Cells from results/replication_params.json.")
    out.append("")
    print("=" * 92)
    print("EOM IDENTITY GATE: exact moments + exact third cumulants -> exact derivatives")
    print("=" * 92)
    allok = True
    for c in cells:
        allok &= run_cell(c, out)
        out.append("")
    verdict = "ALL PASS" if allok else "FAILURE"
    out.append(f"EOM IDENTITY GATE: {verdict}")
    with open(LOG, "w") as f:
        f.write("\n".join(out) + "\n")
    print("=" * 92)
    print(f"EOM IDENTITY GATE: {verdict}")
    print(f"log written to {os.path.relpath(LOG, REPO)}")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
