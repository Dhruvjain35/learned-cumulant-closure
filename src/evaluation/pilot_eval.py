"""
Pilot evaluation (PROTOCOL.md decision table). Trains v3-RAW and v3-RATIO on the IDENTICAL pilot dataset,
then judges per cell -- interpolation cells, extrapolation cells, and an initial-condition stress test --
against exact mesolve references at M=5 with held-out disorder (raw seed 7, never trained).

Methods per cell: plain 2nd-order cumulant | v2 closure (regime-blind, frozen lc_new.pkl) | v3-RAW |
v3-RATIO. Mean-field note: from these initial conditions (all coherences zero) the mean-field equations
are identically frozen (every derivative vanishes at the inverted fixed point) -- superradiance requires
the quantum fluctuations MF discards; stated, not plotted.

Decision rules (pre-registered): interpolation requires v3 (either rep) to beat plain on median NRMSE_n
with 100% stability; representation is chosen on EXTRAPOLATION cells; if BOTH reps lose to plain on
extrapolation cells, STOP (fix representation before the full campaign). Per-cell numbers always shown.
"""
import pickle
import time
import numpy as np
import qutip as qt
from scipy.integrate import solve_ivp

from scalable_engine import ScalableEngine
from closure_v3 import LocalClosureV3, ClosureApplyV3, ratio_transform
from local_run import ClosureApply, observables, stable_ok
from exact_tc import build_ops, build_H, collapse_ops, draw_disorder

WC, GDOWN, NC = 1.0, 0.02, 12
TMAX, NT = 40.0, 200
TEST_SEED = 7

CELLS = [
    ("I1  interp g,k",        0.55, 0.10, 0.02, 0.15),
    ("I2  interp all",        0.85, 0.20, 0.10, 0.20),
    ("I3  center/new seed",   0.70, 0.15, 0.02, 0.15),
    ("E1  high dephasing",    0.70, 0.15, 0.30, 0.15),
    ("E2  high disorder",     0.70, 0.15, 0.02, 0.40),
    ("E3  strong coupling",   1.10, 0.15, 0.02, 0.15),
    ("E4  weak coupling",     0.30, 0.15, 0.02, 0.15),
    ("E5  high kappa",        0.70, 0.40, 0.02, 0.15),
    ("E6  low kappa",         0.70, 0.03, 0.02, 0.15),
]


def build_cell(M, gc, ka, gp, sg, seed):
    w = draw_disorder(M, WC, sg, seed, "gauss")
    g = np.full(M, gc / np.sqrt(M))
    eng = ScalableEngine(M, NC, WC, w, g, kappa=ka, gdown=GDOWN, gphi=gp, Fsafe=6).compile()
    return eng, w, g


def m0_product(eng, f=1.0):
    """Product-state moments: cavity vacuum, each emitter excited with probability f (diagonal mixed for
    f<1). Nonzero only for p=q=0 with all emitter factors in {I, sz}; each sz factor contributes (2f-1)."""
    m = np.zeros(len(eng.track), complex)
    for i, (p, q, es) in enumerate(eng.track):
        if p == 0 and q == 0 and all(e in (0, 3) for e in es):
            m[i] = (2.0 * f - 1.0) ** sum(1 for e in es if e == 3)
    return m


def run_from(eng, m0, closure=None, rtol=1e-4, atol=1e-6):
    ts = np.linspace(0, TMAX, NT)

    def rhs(t, y):
        n = len(y) // 2
        mm = y[:n] + 1j * y[n:]
        kap = closure(mm) if closure is not None else None
        d = eng.deriv_compiled(mm, kap)
        return np.concatenate([d.real, d.imag])

    sol = solve_ivp(rhs, (ts[0], ts[-1]), np.concatenate([m0.real, m0.imag]),
                    t_eval=ts, rtol=rtol, atol=atol, method="RK45")
    n = sol.y.shape[0] // 2
    return ts, np.array([sol.y[:n, i] + 1j * sol.y[n:, i] for i in range(sol.y.shape[1])])


def exact_ref(M, w, g, ka, gp, f=1.0):
    ops = build_ops(M, NC)
    H = build_H(ops, WC, w, g)
    cops = collapse_ops(ops, kappa=ka, gdown=GDOWN, gphi=gp)
    if f >= 1.0:
        rho0 = qt.tensor(qt.basis(NC, 0), *[qt.basis(2, 0) for _ in range(M)])
    else:
        up = f * qt.fock_dm(2, 0) + (1 - f) * qt.fock_dm(2, 1)      # |0> = excited
        rho0 = qt.tensor(qt.fock_dm(NC, 0), *[up for _ in range(M)])
    n = ops["a"].dag() * ops["a"]
    zmean = sum(ops["sz"]) / M
    ts = np.linspace(0, TMAX, NT)
    e = qt.mesolve(H, rho0, ts, c_ops=cops, e_ops=[n, zmean], options={"progress_bar": False}).expect
    ne, ze = np.array(e[0]).real, np.array(e[1]).real
    return ne, ze, float(ne.max())


def nrmse(x, xe):
    L = min(len(x), len(xe))
    return float(np.sqrt(np.mean((x[:L] - xe[:L]) ** 2)) / max(np.sqrt(np.mean(xe[:L] ** 2)), 1e-12))


def cell_row(eng, det, regime, closures, ne, ze):
    out = {}
    m0 = m0_product(eng, 1.0)
    for name, clo in closures.items():
        _, traj = run_from(eng, m0, clo)
        if not stable_ok(traj):
            out[name] = (float("inf"), float("inf"), False)
            continue
        n, z = observables(eng, traj)
        out[name] = (nrmse(n, ne), nrmse(z, ze), True)
    return out


if __name__ == "__main__":
    M = 5
    d = np.load("results/train_v3_pilot.npz", allow_pickle=True)["data"].item()
    lc_raw = LocalClosureV3(); lc_raw.fit(d)
    d_ratio = {t: (ratio_transform(X), Y) for t, (X, Y) in d.items()}
    lc_ratio = LocalClosureV3(); lc_ratio.fit(d_ratio)
    with open("results/lc_new.pkl", "rb") as fh:
        lc2 = pickle.load(fh)
    print("trained v3-RAW and v3-RATIO on identical pilot data; v2 loaded (frozen)\n")
    print("NOTE mean-field: identically frozen from these ICs (all coherences 0 -> every MF derivative 0);")
    print("superradiance needs the fluctuations MF discards. Bottom rung of the hierarchy by physics.\n")

    hdr = f"{'cell':<22s} {'nc/NC':>6s} | {'plain':>7s} {'v2':>7s} {'v3raw':>7s} {'v3rat':>7s}  (NRMSE_n; z in log)"
    print(hdr)
    rows = {}
    for (tag, gc, ka, gp, sg) in CELLS:
        t0 = time.time()
        eng, w, g = build_cell(M, gc, ka, gp, sg, TEST_SEED)
        det = w - WC
        ne, ze, npk = exact_ref(M, w, g, ka, gp)
        guard = "" if npk / NC < 0.5 else "  REF-UNSAFE(nc_peak)"
        closures = {
            "plain": None,
            "v2": ClosureApply(lc2, eng, det),
            "v3raw": ClosureApplyV3(lc_raw, eng, det, (gc, ka, gp)),
            "v3rat": ClosureApplyV3(lc_ratio, eng, det, (gc, ka, gp), xform=ratio_transform),
        }
        r = cell_row(eng, det, (gc, ka, gp), closures, ne, ze)
        rows[tag] = r
        print(f"{tag:<22s} {npk/NC:>6.2f} | {r['plain'][0]:>7.3f} {r['v2'][0]:>7.3f} "
              f"{r['v3raw'][0]:>7.3f} {r['v3rat'][0]:>7.3f}  "
              f"z: {r['plain'][1]:.3f}/{r['v2'][1]:.3f}/{r['v3raw'][1]:.3f}/{r['v3rat'][1]:.3f}"
              f"{guard}  ({time.time()-t0:.0f}s)", flush=True)

    # initial-condition stress: partial inversion f=0.75 at the center regime
    tag, gc, ka, gp, sg = "IC  f=0.75 stress", 0.70, 0.15, 0.02, 0.15
    eng, w, g = build_cell(M, gc, ka, gp, sg, TEST_SEED)
    det = w - WC
    ne, ze, npk = exact_ref(M, w, g, ka, gp, f=0.75)
    m0 = m0_product(eng, 0.75)
    r = {}
    for name, clo in {
        "plain": None, "v2": ClosureApply(lc2, eng, det),
        "v3raw": ClosureApplyV3(lc_raw, eng, det, (gc, ka, gp)),
        "v3rat": ClosureApplyV3(lc_ratio, eng, det, (gc, ka, gp), xform=ratio_transform),
    }.items():
        _, traj = run_from(eng, m0, clo)
        r[name] = (nrmse(observables(eng, traj)[0], ne), nrmse(observables(eng, traj)[1], ze),
                   stable_ok(traj))
    rows[tag] = r
    print(f"{tag:<22s} {npk/NC:>6.2f} | {r['plain'][0]:>7.3f} {r['v2'][0]:>7.3f} "
          f"{r['v3raw'][0]:>7.3f} {r['v3rat'][0]:>7.3f}  "
          f"z: {r['plain'][1]:.3f}/{r['v2'][1]:.3f}/{r['v3raw'][1]:.3f}/{r['v3rat'][1]:.3f}", flush=True)

    # ---- decision table (pre-registered) ----
    icells = [t for t in rows if t.startswith("I")]
    ecells = [t for t in rows if t.startswith("E")]
    med = lambda ts, m: float(np.median([rows[t][m][0] for t in ts]))
    print("\nDECISION TABLE")
    for rep in ("v3raw", "v3rat"):
        interp_ok = med(icells, rep) < med(icells, "plain") and all(rows[t][rep][2] for t in icells)
        e_wins = sum(rows[t][rep][0] < rows[t]["plain"][0] for t in ecells)
        print(f"  {rep}: interpolation {'PASS' if interp_ok else 'FAIL'} "
              f"(median NRMSE_n {med(icells, rep):.3f} vs plain {med(icells, 'plain'):.3f}); "
              f"extrapolation wins {e_wins}/{len(ecells)} cells")
    kill = all(sum(rows[t][rep][0] < rows[t]["plain"][0] for t in ecells) == 0
               for rep in ("v3raw", "v3rat"))
    print(f"  KILL CONDITION (both reps lose ALL extrapolation cells): {'TRIGGERED - STOP' if kill else 'not triggered'}")
    np.savez_compressed("results/pilot_eval.npz", rows=np.array(rows, dtype=object))
    print("saved results/pilot_eval.npz")
