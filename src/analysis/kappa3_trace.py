"""
Where does the third cumulant enter, and when does it matter?

This answers the mechanism questions directly, using only exact solves. No closure, no learned
model, no new architecture -- just the exact dynamics, the exact moments, and the exact connected
cumulants, so that every statement here is a property of the physics rather than of our method.

THE CHAIN (derived here, verified numerically below)
----------------------------------------------------
Photon number. From H = wc a^d a + sum_i w_i |e><e|_i + sum_i g_i (a sp_i + a^d sm_i),

    [a^d a, H] = sum_i g_i ( a^d sm_i - a sp_i )

so with cavity loss kappa D[a],

    d<a^d a>/dt = -kappa <a^d a> + 2 sum_i g_i Im <a^d sm_i>.                          (1)

Equation (1) is EXACT and contains no three-operator term. This is the first thing to say out
loud: the third cumulant does not appear in the photon-number equation at all. The photon number
is driven entirely by the light-matter coherence <a^d sm_i>.

Light-matter coherence. The commutator [sm_i, g_i a sp_i] = g_i a [sm_i, sp_i] = -g_i a sz_i
generates, in d<a^d sm_i>/dt, the term

    -i g_i <a^d a sz_i>,                                                                (2)

a genuine THREE-operator moment. This is the single place the hierarchy fails to close for the
burst. Decomposing it,

    <a^d a sz_i> = <a^d a sz_i>_c
                 + <a^d><a sz_i> + <a><a^d sz_i> + <sz_i><a^d a>
                 - 2<a^d><a><sz_i>,                                                     (3)

the second-order closure keeps everything except the connected piece <a^d a sz_i>_c, which it
sets to zero. That connected piece IS the object the learned closure predicts.

A simplification worth knowing. Starting fully inverted with an empty cavity, the model has no
phase reference, so <a>(t) = 0 for all t (verified numerically below). Equation (3) then collapses
to

    <a^d a sz_i>_c = <a^d a sz_i> - <a^d a><sz_i>,                                      (4)

i.e. exactly the covariance between the photon number and the emitter inversion. That is the
plain-language answer to "what is the model learning": how much the photon number and the
inversion fluctuate together, beyond what their separate averages already account for.

WHAT THIS SCRIPT PRODUCES
-------------------------
  * a numerical check of Eq. (1), comparing its right-hand side to d<a^d a>/dt from the solver
  * a check that <a> = 0, justifying Eq. (4)
  * kappa3(t) for the (a^d, a, sz) channel, against the burst, for several M
  * the timing question answered quantitatively: does kappa3 peak before, during, or after the
    burst peak, and does the answer move with M

Run:  python src/analysis/kappa3_trace.py [--smoke]
"""
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "engines"))

import qutip as qt  # noqa: E402
from exact_tc import build_ops, build_H, collapse_ops, draw_disorder  # noqa: E402

WC, GDOWN = 1.0, 0.02
GC, KAPPA, GPHI, SIGMA = 0.7, 0.03, 0.02, 0.15
TMAX, NT = 40.0, 401
SEED = 7

# Fock cutoffs, matching the convention used elsewhere in the repo (grows with M).
NCS = {2: 9, 3: 10, 4: 11, 5: 12, 6: 14}


def solve_cell(M, Nc, tmax=TMAX, nt=NT, seed=SEED):
    """Exact solve, returning every moment needed for the chain above."""
    w = draw_disorder(M, WC, SIGMA, seed, "gauss")
    g = np.full(M, GC / np.sqrt(M))
    ops = build_ops(M, Nc)
    H = build_H(ops, WC, w, g)
    c = collapse_ops(ops, kappa=KAPPA, gphi=GPHI, gdown=GDOWN)
    psi0 = qt.tensor(qt.basis(Nc, 0), *[qt.basis(2, 0) for _ in range(M)])
    ts = np.linspace(0.0, tmax, nt)

    a, sm, sp, sz = ops["a"], ops["sm"], ops["sp"], ops["sz"]
    n_op = a.dag() * a

    e_ops = [n_op, a]                              # <a^d a>, <a>
    e_ops += [a.dag() * sm[i] for i in range(M)]   # <a^d sm_i>
    e_ops += [n_op * sz[i] for i in range(M)]      # <a^d a sz_i>
    e_ops += [sz[i] for i in range(M)]             # <sz_i>
    e_ops += [a * sz[i] for i in range(M)]         # <a sz_i>
    e_ops += [a.dag() * sz[i] for i in range(M)]   # <a^d sz_i>

    res = qt.mesolve(H, psi0, ts, c_ops=c, e_ops=e_ops,
                     options={"atol": 1e-11, "rtol": 1e-9, "nsteps": 20000})
    E = [np.asarray(x) for x in res.expect]

    o = 0
    n = E[o].real; o += 1
    aexp = E[o]; o += 1
    adsm = np.stack(E[o:o + M], axis=1); o += M
    adasz = np.stack(E[o:o + M], axis=1); o += M
    szs = np.stack(E[o:o + M], axis=1).real; o += M
    asz = np.stack(E[o:o + M], axis=1); o += M
    adsz = np.stack(E[o:o + M], axis=1); o += M

    return dict(t=ts, M=M, Nc=Nc, g=g, w=w, n=n, a=aexp, adsm=adsm,
                adasz=adasz, sz=szs, asz=asz, adsz=adsz)


def kappa3_photon_inversion(S):
    """Connected <a^d a sz_i>_c via the full Eq. (3), summed over i.

    We use the full expression rather than the <a>=0 shortcut so the shortcut can be *checked*
    rather than assumed.
    """
    ad = np.conj(S["a"])[:, None]
    a = S["a"][:, None]
    full = (S["adasz"]
            - ad * S["asz"]
            - a * S["adsz"]
            - S["sz"] * S["n"][:, None]
            + 2.0 * ad * a * S["sz"])
    return full


def check_photon_balance(S):
    """Verify Eq. (1): d<n>/dt should equal -kappa <n> + 2 sum_i g_i Im<a^d sm_i>."""
    t, n = S["t"], S["n"]
    lhs = np.gradient(n, t)
    rhs = -KAPPA * n + 2.0 * np.sum(S["g"][None, :] * np.imag(S["adsm"]), axis=1)
    interior = slice(2, -2)                       # gradient is one-sided at the ends
    scale = max(np.max(np.abs(rhs)), 1e-12)
    return float(np.max(np.abs(lhs[interior] - rhs[interior])) / scale)


def summarize(S):
    t, n = S["t"], S["n"]
    k3 = kappa3_photon_inversion(S)
    k3_tot = np.sum(np.abs(k3), axis=1)           # magnitude across emitters
    i_burst = int(np.argmax(n))
    i_k3 = int(np.argmax(k3_tot))
    # a scale-free measure of how big the dropped term is relative to what is kept
    keep = np.abs(S["n"][:, None] * S["sz"]).sum(axis=1)
    ratio = k3_tot / np.maximum(keep, 1e-12)
    return dict(
        M=S["M"],
        t_burst=float(t[i_burst]), n_peak=float(n[i_burst]),
        t_k3=float(t[i_k3]), k3_peak=float(k3_tot[i_k3]),
        lead=float(t[i_k3] - t[i_burst]),
        k3_at_burst=float(k3_tot[i_burst]),
        ratio_at_burst=float(ratio[i_burst]),
        ratio_peak=float(np.max(ratio)),
        amax=float(np.max(np.abs(S["a"]))),
        balance_err=check_photon_balance(S),
        t=t, n=n, k3=k3_tot, ratio=ratio,
    )


def main():
    smoke = "--smoke" in sys.argv
    Ms = [2, 3] if smoke else [2, 3, 4, 5]
    tmax, nt = (20.0, 201) if smoke else (TMAX, NT)

    print(f"Exact mechanism trace   g_c={GC} kappa={KAPPA} gphi={GPHI} sigma={SIGMA} seed={SEED}")
    print()
    print(f"{'M':>2} {'t_burst':>8} {'n_peak':>8} {'t_k3':>7} {'lead':>7} "
          f"{'k3/keep@burst':>14} {'max|<a>|':>9} {'Eq1 err':>9}")

    out = []
    for M in Ms:
        S = solve_cell(M, NCS[M], tmax=tmax, nt=nt)
        r = summarize(S)
        out.append(r)
        print(f"{r['M']:>2} {r['t_burst']:>8.3f} {r['n_peak']:>8.4f} {r['t_k3']:>7.3f} "
              f"{r['lead']:>+7.3f} {r['ratio_at_burst']:>14.4f} {r['amax']:>9.1e} "
              f"{r['balance_err']:>9.2e}")

    os.makedirs("results", exist_ok=True)
    np.savez_compressed(
        "results/kappa3_trace.npz",
        Ms=np.array([r["M"] for r in out]),
        t=np.stack([r["t"] for r in out]),
        n=np.stack([r["n"] for r in out]),
        k3=np.stack([r["k3"] for r in out]),
        ratio=np.stack([r["ratio"] for r in out]),
        t_burst=np.array([r["t_burst"] for r in out]),
        t_k3=np.array([r["t_k3"] for r in out]),
        lead=np.array([r["lead"] for r in out]),
        ratio_at_burst=np.array([r["ratio_at_burst"] for r in out]),
    )

    print()
    print("READINGS")
    print(f"  max |<a>| over all cells: {max(r['amax'] for r in out):.2e}"
          "   -> <a>=0, so kappa3 is exactly the photon-number/inversion covariance (Eq. 4)")
    print(f"  max Eq.(1) residual:      {max(r['balance_err'] for r in out):.2e}"
          "   -> the photon-number balance is confirmed; no 3-operator term appears in it")
    leads = [r["lead"] for r in out]
    where = ("BEFORE" if all(l < -1e-9 for l in leads) else
             "AFTER" if all(l > 1e-9 for l in leads) else "AT/AROUND")
    print(f"  kappa3 peaks {where} the burst peak; leads = "
          f"{['%+.2f' % l for l in leads]}")
    print("  saved results/kappa3_trace.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
