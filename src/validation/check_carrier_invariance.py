"""
Common-carrier (rotating-frame) invariance audit -- PROTOCOL.md "Publication finalization P2", registered
2026-10-05 before any shifted run. Additive: no engine, closure, router or frozen artifact is modified.

Shift omega_c -> 1 + C and every w_i -> w_i + C (detunings, g_i, kappa, gdown, gphi, draw and initial state
fixed) for C in {0, 9, 99, 999}, and compare against C = 0:
  - exact mesolve (Nc = 16): n(t), mean z(t), burst peak, t_peak, the neutral coherence <a' s-_1>, and the
    charged moments <a>, <s-_1>, <a s-_1> raw and phase-corrected by exp(i C Q t);
  - plain second order and frozen V7 through the unmodified ScalableEngine / run_from / run_v7: the same
    observables, V7 branch, every tracked moment split by excitation-number charge Q, and the kappa3 the
    frozen closure applies along the trajectory (neutral vs charged keys);
  - the lowest layer where the carrier exists: C_shift - C_base must equal i C Q(k) on each tracked row's
    own diagonal entry (ScalableEngine.build_C at M = 2, 3; dense CumulantEngine at M = 2).
Thresholds and the failure diagnostic are the pre-registered ones; they are not tuned here.

    python run.py src/validation/check_carrier_invariance.py
"""
import time
import numpy as np
import qutip as qt

from exact_tc import build_ops, build_H, collapse_ops, draw_disorder
from cumulant import CumulantEngine
from scalable_engine import ScalableEngine
from pilot_eval import m0_product, run_from, WC, GDOWN, NC
from local_run import observables, stable_ok
from v7 import run_v7, route, _load
from closure_v3 import ClosureApplyV3, ratio_transform
from closure_v6 import ClosureApplyV6, run_v6, lambdas

SIGMA, DRAW, NC_REF, FSAFE = 0.15, 50, 16, 6
TMAX, NT = 40.0, 200
SHIFTS = [0.0, 9.0, 99.0, 999.0]
MS = [2, 3]
REGIMES = {"POCKET": (0.70, 0.03, 0.02), "CONTROL": (0.70, 0.15, 0.02)}

# pre-registered thresholds
TOL_EXACT, TOL_EXACT_CHARGED = 1e-5, 1e-7
TOL_CLOS, TOL_CHARGED_CLOS = 1e-4, 1e-12
TOL_BENCH = 1e-4
TOL_LAYER = 1e-10
TOL_REGRESSION = 1e-8
MESOLVE_OPTS = {"progress_bar": False, "nsteps": 1_000_000}
MESOLVE_TIGHT = dict(MESOLVE_OPTS, rtol=1e-9, atol=1e-11)

SIGN = {0: 0, 1: 1, 2: -1, 3: 0}            # emitter code -> charge (s+ raises, s- lowers)


def charge(key):
    p, q, es = key
    return (p - q) + sum(SIGN[e] for e in es)


def cell_freqs(M, C):
    w0 = draw_disorder(M, WC, SIGMA, DRAW, "gauss")
    return WC + C, w0 + C


# ---------------------------------------------------------------- exact
def exact_run(M, gc, ka, gp, C, opts=MESOLVE_OPTS):
    wc, w = cell_freqs(M, C)
    g = np.full(M, gc / np.sqrt(M))
    ops = build_ops(M, NC_REF)
    H = build_H(ops, wc, w, g)
    cops = collapse_ops(ops, kappa=ka, gdown=GDOWN, gphi=gp)
    a, sm = ops["a"], ops["sm"]
    eops = [a.dag() * a, sum(ops["sz"]) / M, a.dag() * sm[0], a, sm[0], a * sm[0]]
    psi0 = qt.tensor(qt.basis(NC_REF, 0), *[qt.basis(2, 0) for _ in range(M)])
    ts = np.linspace(0, TMAX, NT)
    e = qt.mesolve(H, psi0, ts, c_ops=cops, e_ops=eops, options=opts).expect
    return dict(ts=ts, n=np.real(e[0]), z=np.real(e[1]), coh=np.asarray(e[2]),
                ch={"<a>": (np.asarray(e[3]), -1), "<s-_1>": (np.asarray(e[4]), -1),
                    "<a s-_1>": (np.asarray(e[5]), -2)})


# ---------------------------------------------------------------- closures
def build_eng(M, gc, ka, gp, C):
    wc, w = cell_freqs(M, C)
    g = np.full(M, gc / np.sqrt(M))
    eng = ScalableEngine(M, NC, wc, w, g, kappa=ka, gdown=GDOWN, gphi=gp, Fsafe=FSAFE).compile()
    return eng, w - wc


def kappa3_along(eng, det, regime, branch, traj, rtol=1e-4, atol=1e-6):
    """Re-derive the kappa3 the frozen branch applies at each saved point. For V6 the loop's own traces are
    needed, so the branch is re-run directly and its trajectory is asserted identical to V7's."""
    gc, ka, gp = regime
    if branch == "v4":
        clo = ClosureApplyV3(_load("v3dens"), eng, det, regime, xform=ratio_transform)
        return np.array([clo(m) for m in traj])
    clo = ClosureApplyV6(_load("v6"), eng, det, regime, xform=ratio_transform)
    _, t6, h = run_v6(eng, eng_m0(eng), clo, lambdas(ka, gp), rtol=rtol, atol=atol)
    assert np.array_equal(t6, traj), "direct V6 rollout differs from V7's routed rollout"
    out = []
    for m, hh in zip(traj, h):
        clo.current_h = hh
        out.append(clo(m))
    return np.array(out)


def eng_m0(eng):
    return m0_product(eng, 1.0)


def v7_tight(eng, det, regime):
    """Diagnostic only: the same frozen branch V7 selects, at tightened tolerances."""
    gc, ka, gp = regime
    if route(ka, gp) == "v6":
        clo = ClosureApplyV6(_load("v6"), eng, det, regime, xform=ratio_transform)
        _, t, _ = run_v6(eng, eng_m0(eng), clo, lambdas(ka, gp), rtol=1e-8, atol=1e-10)
        return t
    clo = ClosureApplyV3(_load("v3dens"), eng, det, regime, xform=ratio_transform)
    return run_from(eng, eng_m0(eng), clo, rtol=1e-8, atol=1e-10)[1]


def closure_runs(M, regime, C):
    eng, det = build_eng(M, *regime, C)
    m0 = eng_m0(eng)
    _, tp = run_from(eng, m0, None)
    branch, t7 = run_v7(eng, m0, det, regime)
    k3 = kappa3_along(eng, det, regime, branch, t7)
    Q = np.array([charge(k) for k in eng.track])
    Q3 = np.array([charge(k) for k in eng.o3_keys])
    return dict(eng=eng, det=det, plain=tp, v7=t7, branch=branch, k3=k3, Q=Q, Q3=Q3)


# ---------------------------------------------------------------- comparisons
def peak(ts, n):
    i = int(np.argmax(n))
    return float(n[i]), i


def rel_traj(x, x0):
    return float(np.max(np.abs(x - x0)) / max(np.max(np.abs(x0)), 1e-300))


def layer_check(M, regime, C):
    """Lowest layer: coefficient matrices. Returns max |(C_shift - C_base) - i C Q| over all entries."""
    out = {}
    e0, _ = build_eng(M, *regime, 0.0)
    eC, _ = build_eng(M, *regime, C)
    D = eC.build_C() - e0.build_C()
    expect = np.zeros_like(D)
    for a, k in enumerate(e0.track):
        expect[a, e0.idx[k]] = 1j * C * charge(k)
    out["scalable"] = float(np.max(np.abs(D - expect)))
    if M == 2:
        gc, ka, gp = regime
        g = np.full(M, gc / np.sqrt(M))
        wc0, w0 = cell_freqs(M, 0.0)
        wcC, wC = cell_freqs(M, C)
        d0 = CumulantEngine(M, NC, wc0, w0, g, kappa=ka, gdown=GDOWN, gphi=gp, Fsafe=FSAFE)
        dC = CumulantEngine(M, NC, wcC, wC, g, kappa=ka, gdown=GDOWN, gphi=gp, Fsafe=FSAFE)
        Dd = dC.C - d0.C
        expd = np.zeros_like(Dd)
        for a, k in enumerate(d0.track):
            expd[a, d0.idx[k]] = 1j * C * charge(k)
        out["dense"] = float(np.max(np.abs(Dd - expd)))
    return out


def main():
    t0 = time.time()
    L = []
    say = lambda s="": (print(s, flush=True), L.append(s))
    say("COMMON-CARRIER INVARIANCE AUDIT (PROTOCOL.md P2, registered 2026-10-05, commit 78e9a38)")
    say(f"draw {DRAW}, sigma {SIGMA}, gdown {GDOWN}; shifts C = {SHIFTS}; exact Nc={NC_REF}, closures Nc={NC} "
        f"Fsafe={FSAFE}; mesolve {MESOLVE_OPTS}")
    ref = np.load("results/burst_pred_exact.npz", allow_pickle=True)["data"].item()

    rows, fails, diag = [], [], []
    record = {}
    for name, regime in REGIMES.items():
        gc, ka, gp = regime
        for M in MS:
            say(f"\n=== {name} M={M}  (g_c, kappa, gphi) = {regime}  route = {route(ka, gp)} ===")
            base_ex = exact_run(M, gc, ka, gp, 0.0)
            base_cl = closure_runs(M, regime, 0.0)
            ts = base_ex["ts"]
            nE0, iE0 = peak(ts, base_ex["n"])
            n_p0, _ = observables(base_cl["eng"], base_cl["plain"])
            n_v0, _ = observables(base_cl["eng"], base_cl["v7"])
            pP0, iP0 = peak(ts, n_p0)
            pV0, iV0 = peak(ts, n_v0)
            errP0, errV0 = (pP0 - nE0) / nE0, (pV0 - nE0) / nE0

            # regression against the committed headline run (same draw, same code path)
            r = ref[(name, M, DRAW)][0]
            reg = {"exact": abs(ka * nE0 - r["ref"]["Fpeak"]) / r["ref"]["Fpeak"],
                   "plain": abs(ka * pP0 - r["plain"]["Fpeak"]) / r["plain"]["Fpeak"],
                   "v7": abs(ka * pV0 - r["v7"]["Fpeak"]) / r["v7"]["Fpeak"]}
            regok = all(v <= TOL_REGRESSION for v in reg.values())
            say(f"  C=0 regression vs burst_pred_exact.npz (rel. peak): " +
                "  ".join(f"{k} {v:.1e}" for k, v in reg.items()) + f"   {'PASS' if regok else 'FAIL'}")
            say(f"  C=0 signed burst-peak error: plain {errP0:+.5f}   V7 {errV0:+.5f}   (branch {base_cl['branch']})")
            if not regok:
                fails.append((name, M, 0.0, "regression"))

            for C in SHIFTS[1:]:
                ex = exact_run(M, gc, ka, gp, C)
                cl = closure_runs(M, regime, C)
                res = {}
                # exact
                nE, iE = peak(ts, ex["n"])
                res["ex_n"] = rel_traj(ex["n"], base_ex["n"])
                res["ex_z"] = float(np.max(np.abs(ex["z"] - base_ex["z"])))
                res["ex_peak"] = abs(nE - nE0) / nE0
                res["ex_tpk"] = int(iE != iE0)
                res["ex_coh"] = rel_traj(ex["coh"], base_ex["coh"])
                res["ex_charged_raw"] = max(float(np.max(np.abs(v))) for v, _ in ex["ch"].values())
                res["ex_charged_phase"] = max(
                    float(np.max(np.abs(v - np.exp(1j * C * Qk * ts) * base_ex["ch"][k][0])))
                    for k, (v, Qk) in ex["ch"].items())
                # closures
                n_p, z_p = observables(cl["eng"], cl["plain"])
                n_v, z_v = observables(cl["eng"], cl["v7"])
                _, z_p0 = observables(base_cl["eng"], base_cl["plain"])
                _, z_v0 = observables(base_cl["eng"], base_cl["v7"])
                pP, iP = peak(ts, n_p)
                pV, iV = peak(ts, n_v)
                neu = base_cl["Q"] == 0
                for tag, tr, tr0, nn, nn0, zz, zz0, pk, pk0, ii, ii0 in (
                        ("pl", cl["plain"], base_cl["plain"], n_p, n_p0, z_p, z_p0, pP, pP0, iP, iP0),
                        ("v7", cl["v7"], base_cl["v7"], n_v, n_v0, z_v, z_v0, pV, pV0, iV, iV0)):
                    res[f"{tag}_stable"] = int(stable_ok(tr))
                    res[f"{tag}_n"] = rel_traj(nn, nn0)
                    res[f"{tag}_z"] = float(np.max(np.abs(zz - zz0)))
                    res[f"{tag}_peak"] = abs(pk - pk0) / pk0
                    res[f"{tag}_tpk"] = int(ii != ii0)
                    res[f"{tag}_neutral"] = rel_traj(tr[:, neu], tr0[:, neu])
                    res[f"{tag}_charged"] = float(np.max(np.abs(tr[:, ~neu])))
                res["v7_branch_same"] = int(cl["branch"] == base_cl["branch"])
                n3 = base_cl["Q3"] == 0
                res["k3_neutral"] = rel_traj(cl["k3"][:, n3], base_cl["k3"][:, n3])
                res["k3_charged"] = float(np.max(np.abs(cl["k3"][:, ~n3]))) if (~n3).any() else 0.0
                res["det_resid"] = float(np.max(np.abs(cl["det"] - base_cl["det"])))
                errP, errV = (pP - nE) / nE, (pV - nE) / nE
                res["bench_plain"] = abs(errP - errP0)
                res["bench_v7"] = abs(errV - errV0)
                res.update({f"layer_{k}": v for k, v in layer_check(M, regime, C).items()})

                checks = {
                    "ex_n": res["ex_n"] <= TOL_EXACT, "ex_z": res["ex_z"] <= TOL_EXACT,
                    "ex_peak": res["ex_peak"] <= TOL_EXACT, "ex_tpk": res["ex_tpk"] == 0,
                    "ex_coh": res["ex_coh"] <= TOL_EXACT,
                    "ex_charged": res["ex_charged_raw"] <= TOL_EXACT_CHARGED
                    and res["ex_charged_phase"] <= TOL_EXACT_CHARGED,
                    "branch": res["v7_branch_same"] == 1,
                    "k3_neutral": res["k3_neutral"] <= TOL_CLOS,
                    "k3_charged": res["k3_charged"] <= TOL_CHARGED_CLOS,
                    "bench_plain": res["bench_plain"] <= TOL_BENCH, "bench_v7": res["bench_v7"] <= TOL_BENCH,
                }
                for tag in ("pl", "v7"):
                    checks[f"{tag}_stable"] = res[f"{tag}_stable"] == 1
                    for m in ("n", "z", "peak", "neutral"):
                        checks[f"{tag}_{m}"] = res[f"{tag}_{m}"] <= TOL_CLOS
                    checks[f"{tag}_tpk"] = res[f"{tag}_tpk"] == 0
                    checks[f"{tag}_charged"] = res[f"{tag}_charged"] <= TOL_CHARGED_CLOS
                for k, v in res.items():
                    if k.startswith("layer_"):
                        checks[k] = v <= TOL_LAYER * max(1.0, C)
                bad = [k for k, ok in checks.items() if not ok]

                say(f"  C={C:>5.0f}  wc={1 + C:>6.0f}  g_c/wc={gc / (1 + C):.1e}  |det_C - det_0|={res['det_resid']:.1e}")
                say(f"     exact : n {res['ex_n']:.1e}  z {res['ex_z']:.1e}  peak {res['ex_peak']:.1e}  "
                    f"tpk-shift {res['ex_tpk']}  <a's-_1> {res['ex_coh']:.1e}  "
                    f"charged raw {res['ex_charged_raw']:.1e} / phase-corr {res['ex_charged_phase']:.1e}")
                for tag, lab in (("pl", "plain "), ("v7", "V7    ")):
                    say(f"     {lab}: n {res[tag + '_n']:.1e}  z {res[tag + '_z']:.1e}  peak {res[tag + '_peak']:.1e}  "
                        f"tpk-shift {res[tag + '_tpk']}  neutral moments {res[tag + '_neutral']:.1e}  "
                        f"charged max {res[tag + '_charged']:.1e}  stable {res[tag + '_stable']}")
                say(f"     V7 kappa3: neutral keys {res['k3_neutral']:.1e}  charged keys max {res['k3_charged']:.1e}  "
                    f"branch {cl['branch']} (same {res['v7_branch_same']})")
                say(f"     benchmark |d signed err|: plain {res['bench_plain']:.1e}  V7 {res['bench_v7']:.1e}   "
                    f"(signed now plain {errP:+.5f}, V7 {errV:+.5f})")
                say("     layer |dC - iCQ|: " + "  ".join(f"{k[6:]} {v:.1e}" for k, v in res.items()
                                                       if k.startswith("layer_")))
                say(f"     -> {'PASS' if not bad else 'EXCEEDS: ' + ', '.join(bad)}")

                if bad:
                    # pre-registered single diagnostic: tightened tolerances at C=0 and this C
                    dg = {}
                    if any(b.startswith("ex_") for b in bad):
                        e0t = exact_run(M, gc, ka, gp, 0.0, MESOLVE_TIGHT)
                        eCt = exact_run(M, gc, ka, gp, C, MESOLVE_TIGHT)
                        dg["exact_n"] = (res["ex_n"], rel_traj(eCt["n"], e0t["n"]))
                    if any(b.startswith(("pl_", "v7_", "bench", "k3")) for b in bad):
                        eng0, det0 = base_cl["eng"], base_cl["det"]
                        engC, detC = cl["eng"], cl["det"]
                        p0t = run_from(eng0, eng_m0(eng0), None, rtol=1e-8, atol=1e-10)[1]
                        pCt = run_from(engC, eng_m0(engC), None, rtol=1e-8, atol=1e-10)[1]
                        v0t, vCt = v7_tight(eng0, det0, regime), v7_tight(engC, detC, regime)
                        dg["plain_n"] = (res["pl_n"], rel_traj(observables(engC, pCt)[0],
                                                              observables(eng0, p0t)[0]))
                        dg["v7_n"] = (res["v7_n"], rel_traj(observables(engC, vCt)[0],
                                                           observables(eng0, v0t)[0]))
                    noise = all(after <= before / 10 for before, after in dg.values()) if dg else False
                    say("     DIAGNOSTIC (tight tol): " + "  ".join(f"{k} {b:.1e}->{a:.1e}" for k, (b, a) in dg.items())
                        + f"  -> {'integrator-tolerance noise (PASS after tolerance diagnostic)' if noise else 'GENUINE: FAIL'}")
                    diag.append((name, M, C, bad, dg, noise))
                    if not noise:
                        fails.append((name, M, C, ",".join(bad)))
                record[(name, M, C)] = res
                rows.append((name, M, C, res))

    # overall
    keys = sorted({k for _, _, _, r in rows for k in r})
    worst = {k: max(r[k] for _, _, _, r in rows if k in r) for k in keys}   # layer_dense exists at M=2 only
    say("\n=== MAXIMUM OVER ALL (regime, M, C>0) ===")
    for k in keys:
        say(f"  {k:<18s} {worst[k]:.2e}")
    clean = not fails and not diag
    verdict = "PASS" if clean else ("PASS after tolerance diagnostic" if not fails else "FAIL")
    say(f"\nOVERALL VERDICT: {verdict}" + ("" if not fails else f"   failures: {fails}"))
    say(f"({time.time() - t0:.0f}s)")
    np.savez_compressed("results/carrier_invariance.npz", record=np.array(record, dtype=object),
                        worst=np.array(worst, dtype=object), shifts=np.array(SHIFTS),
                        verdict=np.array(verdict), diagnostics=np.array(diag, dtype=object))
    with open("results/carrier_invariance.log", "w") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
