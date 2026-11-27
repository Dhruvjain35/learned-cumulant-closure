"""Exact quantum dynamics behind the rendered visuals: photon-number distribution and per-atom excitation.

For the README's showcase cell (3 atoms, low loss, disorder draw 50) and held-out cell H11 (5 atoms), solve the
exact master equation with the same model and cutoff as the burst-peak references (QuTiP mesolve, Nc = 16) and
record, on a fine time grid:
  P(n, t)    photon-number distribution of the cavity (its reduced state is diagonal in the Fock basis here,
             because every term in the model conserves or lowers total excitation and the start state has a
             definite excitation number, so P(n, t) is the whole cavity state)
  p_i(t)     excitation probability of each atom, (1 + <sz_i>) / 2
  n(t)       mean photon number (checked against the burst-peak reference)

    python run.py scripts/export_quantum_states.py

Writes results/quantum_states_{showcase,H11}.npz and demo/data/scene_{showcase,H11}.json. About a minute.
"""
import json
import numpy as np
import qutip as qt

from exact_tc import build_ops, build_H, collapse_ops, draw_disorder

WC, GDOWN, NC = 1.0, 0.02, 16
TS = np.linspace(0.0, 15.0, 301)
CELLS = {
    "showcase": dict(M=3, g_c=0.7, kappa=0.03, gamma_phi=0.02, sigma=0.15, seed=50),
}


def solve(M, g_c, kappa, gamma_phi, sigma, seed):
    w = draw_disorder(M, WC, sigma, seed, "gauss")
    g = np.full(M, g_c / np.sqrt(M))
    ops = build_ops(M, NC)
    H = build_H(ops, WC, w, g)
    cops = collapse_ops(ops, kappa=kappa, gdown=GDOWN, gphi=gamma_phi)
    psi0 = qt.tensor(qt.basis(NC, 0), *[qt.basis(2, 0) for _ in range(M)])
    eye = [qt.qeye(2) for _ in range(M)]
    proj = [qt.tensor(qt.fock_dm(NC, k), *eye) for k in range(NC)]
    n = ops["a"].dag() * ops["a"]
    e = qt.mesolve(H, psi0, TS, c_ops=cops, e_ops=[n] + proj + list(ops["sz"]),
                   options={"progress_bar": False}).expect
    e = [np.real(np.asarray(x)) for x in e]
    nt, P, sz = e[0], np.array(e[1:1 + NC]).T, np.array(e[1 + NC:]).T
    assert np.allclose(P.sum(1), 1, atol=1e-8), "Fock populations do not sum to 1"
    assert np.allclose(P @ np.arange(NC), nt, atol=1e-8), "P(n) inconsistent with <n>"
    return w - WC, nt, P, (1 + sz) / 2


def main():
    spec = json.load(open("results/heldout_param_cells.json"))
    h11 = next(c for c in spec["cells"] if c["id"] == "H11")
    CELLS["H11"] = dict(M=5, g_c=h11["g_c"], kappa=h11["kappa"], gamma_phi=h11["gamma_phi"], sigma=h11["sigma"],
                        seed=h11["seed"])
    ref = {c["id"]: c for c in json.load(open("demo/data/heldout.json"))["cells"]}
    for name, c in CELLS.items():
        det, nt, P, pe = solve(**c)
        if name == "H11":   # same physics as the burst-peak reference: the peaks must agree
            r = ref["H11"]
            print(f"H11 peak n: {nt.max():.5f} here vs {r['exact']['npeak']:.5f} in the reference (coarser grid)")
        np.savez_compressed(f"results/quantum_states_{name}.npz", t=TS, n=nt, P=P, p_exc=pe, detunings=det,
                            params=json.dumps(c))
        r5 = lambda a: [float(f"{x:.4g}") for x in a]
        json.dump(dict(params=c, detunings=r5(det), t=r5(TS), n=r5(nt),
                       p_exc=[r5(pe[:, i]) for i in range(c["M"])], P=[r5(P[:, k]) for k in range(13)]),
                  open(f"demo/data/scene_{name}.json", "w"), separators=(",", ":"))
        tp = TS[np.argmax(nt)]
        print(f"{name}: M={c['M']} peak <n>={nt.max():.3f} at t={tp:.2f}; P(n) max cutoff weight {P[:, -1].max():.2e}")


if __name__ == "__main__":
    main()
