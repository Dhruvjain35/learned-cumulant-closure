"""
FINAL FROZEN MECHANISM ANALYSIS (PROTOCOL "MECHANISM ANALYSIS PRE-REGISTRATION").
Is the exact superradiant burst-peak correction driven by ONE dominant connected-kappa3 family, a small set,
or broadly distributed? No retrain / tune / reroute / new variant. Frozen V7, frozen engines. Ablations zero a
kappa3 family on the CLOSURE OUTPUT only.

kappa3_exact(k,t) = <k>_exact - v3_factorized(m2_exact)   (connected cumulant; GATE: ~0 at t=0).
FAMILY = canonical_sig(k)[0] (conjugation-canonical factor-type signature; conjugation-closed -> Hermiticity).
DRIVER E_f(t) = sum_i Im( T3[row(a sigma^+_i)] . kappa3_masked_to_f )   (contribution to collective
emission-coherence buildup; d<a^dag a>/dt has no direct kappa3 term, so this is the relevant one-level-up rate).
"""
import sys
import pickle
import time
import numpy as np
import qutip as qt
from scipy.integrate import solve_ivp

from pilot_eval import build_cell, m0_product, run_from, WC
from closure_v3 import ClosureApplyV3, ratio_transform, conj_key, canonical_sig, context_v3
from closure_v6 import ClosureApplyV6, run_v6, lambdas, trace_series
from v7 import route
from local_run import observables, stable_ok
from cumulant import closure_terms, mono_matrix
from exact_tc import build_ops, build_H, collapse_ops, draw_disorder
from t3_truncation import T3Engine
from burst_pred import burst_obs, SG, NC_REF

GDOWN = 0.02
TMAX, NT = 40.0, 200
SEEDS = [37, 38, 39]
REGIMES = {"POCKET": (0.70, 0.03, 0.02), "CONTROL": (0.70, 0.15, 0.02)}
FLUX_FRAC = 0.5
SMOKE = "--smoke" in sys.argv


def load(n):
    with open(f"results/lc_{n}.pkl", "rb") as fh:
        return pickle.load(fh)


class AblatedClosure:
    """Zero a kappa3 family on the closure OUTPUT (frozen weights untouched). Forwards current_h to the inner
    applier so it works inside run_v6 as well as run_from."""
    def __init__(self, inner, mask_idx):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_mask", np.asarray(mask_idx, int))

    @property
    def current_h(self):
        return self._inner.current_h

    @current_h.setter
    def current_h(self, v):
        self._inner.current_h = v

    def __call__(self, m):
        kap = np.array(self._inner(m), complex, copy=True)
        kap[self._mask] = 0.0
        return kap


def exact_moment_traj(M, w, g, ka, gp, track_keys, o3_keys, Nc):
    """Exact mesolve; return ts, m2[NT,ntrack], raw3[NT,no3], n(t) (=<a^dag a>), nc_peak_frac."""
    O = build_ops(M, Nc)
    H = build_H(O, WC, w, g)
    cops = collapse_ops(O, kappa=ka, gdown=GDOWN, gphi=gp)
    psi0 = qt.tensor(qt.basis(Nc, 0), *[qt.basis(2, 0) for _ in range(M)])
    e_track = [mono_matrix(k, O) for k in track_keys]
    e_o3 = [mono_matrix(k, O) for k in o3_keys]
    ts = np.linspace(0, TMAX, NT)
    res = qt.mesolve(H, psi0, ts, c_ops=cops, e_ops=e_track + e_o3, options={"progress_bar": False})
    ex = res.expect
    m2 = np.array(ex[:len(track_keys)]).T                      # (NT, ntrack), complex
    raw3 = np.array(ex[len(track_keys):]).T                    # (NT, no3), complex
    nidx = track_keys.index((1, 1, tuple(0 for _ in range(M))))
    n = m2[:, nidx].real
    return ts, m2, raw3, n, float(n.max()) / Nc


def t3_kappa3_traj(M, w, g, ka, gp, o3_keys, eng, Nc_t3=12):
    """Closed-loop T3 (kappa4=0): return kappa3_T3[NT,no3] mapped to eng.o3_keys, or None if unstable."""
    e3 = T3Engine(M, Nc_t3, WC, w, g, kappa=ka, gdown=GDOWN, gphi=gp, Fsafe=6)
    m0 = np.zeros(len(e3.track), complex)
    for i, (p, q, es) in enumerate(e3.track):
        if p == 0 and q == 0 and all(x in (0, 3) for x in es):
            m0[i] = 1.0
    ts = np.linspace(0, TMAX, NT)

    def rhs(t, y):
        nn = len(y) // 2
        mm = y[:nn] + 1j * y[nn:]
        d = e3.deriv(mm)
        return np.concatenate([d.real, d.imag])

    def blow(t, y):
        return 1.0e4 - float(np.max(np.abs(y)))
    blow.terminal = True; blow.direction = -1
    sol = solve_ivp(rhs, (0, TMAX), np.concatenate([m0.real, m0.imag]), t_eval=ts,
                    rtol=1e-5, atol=1e-7, method="LSODA", events=blow, max_step=0.5)
    if not sol.success or sol.t[-1] < TMAX - 1e-6 or not np.all(np.isfinite(sol.y)):
        return None
    nn = len(e3.track)
    traj = (sol.y[:nn] + 1j * sol.y[nn:]).T                    # (NT, nT3track)
    # raw 3rd moments for eng.o3_keys, and m2 over eng.track, from the T3 trajectory
    t3idx = {k: i for i, k in enumerate(e3.track)}
    raw3 = np.array([[traj[ti, t3idx[k]] for k in o3_keys] for ti in range(NT)])
    m2 = np.array([[traj[ti, t3idx[k]] for k in eng.track] for ti in range(NT)])
    kap = np.array([raw3[ti] - eng.v3_factorized(m2[ti]) for ti in range(NT)])
    return kap


def kappa3_of(eng, m2_row, raw3_row):
    return raw3_row - eng.v3_factorized(m2_row)


def emission_rows(eng, M):
    """tracked-moment row indices of <a sigma^+_i> = (0,1, es with es[i]=SPLUS=1)."""
    rows = []
    for i in range(M):
        es = tuple(1 if s == i else 0 for s in range(M))
        rows.append(eng.tpos[(0, 1, es)])
    return np.array(rows, int)


def E_family(eng, erows, kap_vec, mask):
    """Direct kappa3 contribution to collective emission-coherence buildup (pre-registered driver). NOTE:
    structurally only the (ad,a,sz) photon-number-inversion family enters d<a sigma^+>/dt at order 3, so this
    is the DIRECT channel, not an unbiased scan -- the non-circular adjudicators are D_family and the ablation."""
    kap = np.zeros_like(kap_vec); kap[mask] = kap_vec[mask]
    return float(np.sum(np.imag(eng.T3[erows].dot(kap))))


def D_family(eng, kap_vec, mask):
    """Total dynamical influence of family f's kappa3: ||T3 . kappa3_masked_to_f|| over ALL tracked
    derivatives (non-circular; every family can contribute)."""
    kap = np.zeros_like(kap_vec); kap[mask] = kap_vec[mask]
    return float(np.linalg.norm(eng.T3.dot(kap)))


def build_families(eng):
    """Group o3 indices by canonical_sig (conjugation-INVARIANT). o3_keys need not be conjugation-closed (the
    validated EOM only requests the order-3 moments its order-2 derivatives need); but because canonical_sig is
    conjugation-invariant, any conjugate pair that IS present shares a family and is zeroed together -> ablating
    a whole family preserves the closure's Hermiticity (0 = conj 0), and tracked conjugate moments stay paired."""
    fam2idx = {}
    o3set = set(eng.o3_keys)
    for i, k in enumerate(eng.o3_keys):
        fam2idx.setdefault(canonical_sig(k)[0], []).append(i)
        ck = conj_key(k)
        if ck in o3set:                                            # if both present, same family (Hermiticity)
            assert canonical_sig(ck)[0] == canonical_sig(k)[0]
    return {f: np.array(v, int) for f, v in fam2idx.items()}


def fam_label(f):
    return "".join({"ad": "A", "a": "a", "sp": "+", "sm": "-", "sz": "z"}[t] for t in f)


def run_cell(M, regime, gc, ka, gp, seed, v4, v6, want_t3=True):
    eng, w, g = build_cell(M, gc, ka, gp, SG, seed)
    det = w - WC
    fam2idx = build_families(eng)
    erows = emission_rows(eng, M)
    dt = TMAX / (NT - 1)

    ts, m2ex, raw3ex, nex, frac = exact_moment_traj(M, w, g, ka, gp, eng.track, eng.o3_keys, NC_REF)
    kap_ex = np.array([kappa3_of(eng, m2ex[t], raw3ex[t]) for t in range(NT)])
    gate0 = float(np.max(np.abs(kap_ex[0])))                                     # GATE: ~0 at t=0

    flux = ka * nex
    win = np.where(flux >= FLUX_FRAC * flux.max())[0]
    fpk_ex = burst_obs(ts, nex, ka)["Fpeak"]

    # teacher-forced V7 kappa3 (feed exact m2; for V6 feed exact-history traces)
    br = route(ka, gp)
    if br == "v6":
        clo_tf = ClosureApplyV6(v6, eng, det, (gc, ka, gp), xform=ratio_transform)
        ctx_series = np.array([context_v3({k: m2ex[t][j] for j, k in enumerate(eng.track)}, M) for t in range(NT)])
        h_series = trace_series(ctx_series, lambdas(ka, gp), dt)
    else:
        clo_tf = ClosureApplyV3(v4, eng, det, (gc, ka, gp), xform=ratio_transform)
        h_series = None
    kap_v7_tf = np.zeros_like(kap_ex)
    for t in range(NT):
        if h_series is not None:
            clo_tf.current_h = h_series[t]
        kap_v7_tf[t] = clo_tf(m2ex[t])

    # closed-loop V7 kappa3 (its own trajectory) + burst-peak error
    if br == "v6":
        clo_cl = ClosureApplyV6(v6, eng, det, (gc, ka, gp), xform=ratio_transform)
        _, traj7, h7 = run_v6(eng, m0_product(eng, 1.0), clo_cl, lambdas(ka, gp))
        kap_v7_cl = np.zeros_like(kap_ex)
        for t in range(NT):
            clo_cl.current_h = h7[t]
            kap_v7_cl[t] = clo_cl(traj7[t])
    else:
        clo_cl = ClosureApplyV3(v4, eng, det, (gc, ka, gp), xform=ratio_transform)
        _, traj7 = run_from(eng, m0_product(eng, 1.0), clo_cl)
        kap_v7_cl = np.array([clo_cl(traj7[t]) for t in range(NT)])
    n7 = observables(eng, traj7)[0]
    err_v7 = abs(burst_obs(ts, n7, ka)["Fpeak"] - fpk_ex) / max(fpk_ex, 1e-12)

    # plain burst-peak error
    _, trajp = run_from(eng, m0_product(eng, 1.0), None)
    err_plain = abs(burst_obs(ts, observables(eng, trajp)[0], ka)["Fpeak"] - fpk_ex) / max(fpk_ex, 1e-12)

    # T3 closed-loop kappa3 (may be None)
    kap_t3 = t3_kappa3_traj(M, w, g, ka, gp, eng.o3_keys, eng) if want_t3 else None

    # (B) teacher-forced family decomposition over burst window: exact and V7, both E_f (direct channel) and
    #     D_f (total dynamical influence, non-circular)
    fams = list(fam2idx)
    Eex = {f: np.array([E_family(eng, erows, kap_ex[t], fam2idx[f]) for t in win]) for f in fams}
    Ev7 = {f: np.array([E_family(eng, erows, kap_v7_tf[t], fam2idx[f]) for t in win]) for f in fams}
    Dex = {f: float(np.mean([D_family(eng, kap_ex[t], fam2idx[f]) for t in win])) for f in fams}
    Dv7 = {f: float(np.mean([D_family(eng, kap_v7_tf[t], fam2idx[f]) for t in win])) for f in fams}
    score_E = {f: float(np.mean(np.abs(Eex[f]))) for f in fams}
    score_D = dict(Dex)

    # (C) ablation: zero each family in closed-loop V7 -> burst-peak error. Instability on removal = the family
    #     is critical (report frac_lost = inf, i.e. removing it breaks V7).
    frac_lost, abl_unstable = {}, {}
    denom = err_plain - err_v7
    for f in fams:
        if br == "v6":
            abl = AblatedClosure(ClosureApplyV6(v6, eng, det, (gc, ka, gp), xform=ratio_transform), fam2idx[f])
            _, tabl, _ = run_v6(eng, m0_product(eng, 1.0), abl, lambdas(ka, gp))
        else:
            abl = AblatedClosure(ClosureApplyV3(v4, eng, det, (gc, ka, gp), xform=ratio_transform), fam2idx[f])
            _, tabl = run_from(eng, m0_product(eng, 1.0), abl)
        if not stable_ok(tabl):
            frac_lost[f] = np.inf; abl_unstable[f] = True; continue
        abl_unstable[f] = False
        err_abl = abs(burst_obs(ts, observables(eng, tabl)[0], ka)["Fpeak"] - fpk_ex) / max(fpk_ex, 1e-12)
        frac_lost[f] = float((err_abl - err_v7) / denom) if abs(denom) > 1e-9 else np.nan

    return dict(ts=ts, win=win, fams=fams, fam2idx={f: fam2idx[f] for f in fams},
                Eex=Eex, Ev7=Ev7, Dex=Dex, Dv7=Dv7, score_E=score_E, score_D=score_D,
                frac_lost=frac_lost, abl_unstable=abl_unstable,
                kap_ex=kap_ex, kap_v7_cl=kap_v7_cl, kap_v7_tf=kap_v7_tf, kap_t3=kap_t3, erows=erows,
                gate0=gate0, frac=frac, br=br, err_v7=err_v7, err_plain=err_plain,
                eng_o3=eng.o3_keys)


if __name__ == "__main__":
    Ms = {"POCKET": (3 if SMOKE else 4), "CONTROL": (2 if SMOKE else 4)}
    if SMOKE:
        SEEDS = [37]
    v4, v6 = load("v3dens"), load("v6")
    print(f"MECHANISM {'(SMOKE) ' if SMOKE else ''}seeds={SEEDS}; cells POCKET M{Ms['POCKET']}, "
          f"CONTROL M{Ms['CONTROL']}; NC_ref={NC_REF}\n")
    t0 = time.time()
    allres = {}
    for regime, (gc, ka, gp) in REGIMES.items():
        M = Ms[regime]
        for sd in SEEDS:
            r = run_cell(M, regime, gc, ka, gp, sd, v4, v6)
            allres[(regime, sd)] = r
            assert r["gate0"] < 1e-6, f"GATE FAIL kappa3_exact(0)={r['gate0']:.2e} at {regime} seed {sd}"
        print(f"[{regime} M{M}] gate0<1e-6 OK; branch={allres[(regime,SEEDS[0])]['br']}; "
              f"err_plain~{np.mean([allres[(regime,s)]['err_plain'] for s in SEEDS]):.3f} "
              f"err_v7~{np.mean([allres[(regime,s)]['err_v7'] for s in SEEDS]):.3f}", flush=True)

        # aggregate family rankings across seeds: E (direct channel), D (total influence), ablation
        fams = allres[(regime, SEEDS[0])]["fams"]
        totE = {f: np.mean([allres[(regime, s)]["score_E"][f] for s in SEEDS]) for f in fams}
        totD = {f: np.mean([allres[(regime, s)]["score_D"][f] for s in SEEDS]) for f in fams}
        SE = sum(totE.values()) or 1.0; SD = sum(totD.values()) or 1.0
        rankedD = sorted(fams, key=lambda f: -totD[f])                       # rank by NON-CIRCULAR D_f
        print(f"  family  |  E_share  D_share  frac_lost(abl)  V7mag  V7sign   (ranked by D_f, top 8)")
        for f in rankedD[:8]:
            fls = [allres[(regime, s)]["frac_lost"].get(f, np.nan) for s in SEEDS]
            fl = np.nanmean([x for x in fls if np.isfinite(x)]) if any(np.isfinite(x) for x in fls) else np.nan
            ncrit = sum(np.isinf(x) for x in fls)                            # seeds where removal destabilized V7
            rat, sgn = [], []
            for s in SEEDS:
                a = allres[(regime, s)]["Eex"][f]; b = allres[(regime, s)]["Ev7"][f]
                if np.mean(np.abs(a)) > 1e-12:
                    rat.append(np.mean(np.abs(b)) / np.mean(np.abs(a))); sgn.append(np.sign(np.sum(a * b)))
            crit = f" CRIT×{ncrit}" if ncrit else ""
            print(f"   {fam_label(f):7s} | {totE[f]/SE:6.2f}  {totD[f]/SD:6.2f}  {fl:+13.2f}  "
                  f"{np.mean(rat) if rat else float('nan'):5.2f}  {np.mean(sgn) if sgn else float('nan'):+5.1f}{crit}",
                  flush=True)
        allres[(regime, "rankedD")] = rankedD
        allres[(regime, "shareE")] = {f: totE[f] / SE for f in fams}
        allres[(regime, "shareD")] = {f: totD[f] / SD for f in fams}
    np.savez_compressed("results/mechanism.npz", res=np.array(allres, dtype=object))
    print(f"\n({time.time()-t0:.0f}s) saved results/mechanism.npz")
