"""
Equivalence gates: the portable V7 package (portable_v7/pv7, NumPy only) against the frozen originals (sklearn
pickles + src/ code). Software equivalence, not science: the Result 40 cells used here were chosen because they
span the support classes and both branches, and they are NOT the cells reserved for the external panel.

Tolerances, fixed before the first run (same machine, so differences should be at rounding level or zero):
  T_FEAT   = 0        features must be bit-identical (same arithmetic on the same inputs)
  T_STATIC = 1e-12    max |pickle prediction - portable prediction|, every network, every type
  T_CLOS   = 1e-12    max |original kappa3 - portable kappa3| over every order-3 key
  T_TRAJ   = 1e-10    max |original - portable| over every tracked moment of full rollouts
Plus exact checks: router agreement on a boundary grid, vendored engine identity, clip-bound identity, the
feature-spec ratio columns, conjugate and self-conjugate structure.

    python run.py tests/test_portable_v7.py          (writes results/portable_v7_equivalence.log)
"""
import os
import sys
import pickle
import warnings
import numpy as np

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "portable_v7"))

import pv7.features as pf                                         # noqa: E402
import pv7.rollout as pr                                          # noqa: E402
from pv7.closure import ClosureApply as PClosure                  # noqa: E402
from pv7.router import route as proute                            # noqa: E402

import closure_v3 as o3                                           # noqa: E402
import closure_v6 as o6                                           # noqa: E402
import local_closure as olc                                       # noqa: E402
import v7 as ov7                                                  # noqa: E402
import pilot_eval as ope                                          # noqa: E402
from local_run import observables as oobs                         # noqa: E402

T_FEAT, T_STATIC, T_CLOS, T_TRAJ = 0.0, 1e-12, 1e-12, 1e-10
CELLS = [  # id, g_c, kappa, gamma_phi, sigma, seed, class, expected branch   (Result 40, NOT the external panel)
    ("H02", 0.85, 0.22, 0.04, 0.20, 301, "B", "v4"), ("H10", 0.70, 0.045, 0.045, 0.10, 309, "B", "v6"),
    ("H16", 0.70, 0.15, 0.12, 0.15, 315, "C", "v4"), ("H15", 0.70, 0.04, 0.015, 0.15, 314, "C", "v6"),
    ("H26", 0.35, 0.10, 0.015, 0.20, 325, "D", "v4"), ("H29", 0.70, 0.025, 0.04, 0.38, 328, "D", "v6"),
]
FIXTURE = [("FIX-V4", 2, 0.70, 0.15, 0.02, 0.15, 50, "v4"), ("FIX-V6", 2, 0.70, 0.03, 0.02, 0.15, 50, "v6")]
L = []
FAIL = []


def say(s=""):
    print(s, flush=True)
    L.append(s)


def check(name, ok, detail=""):
    say(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAIL.append(name)


def orig_closure(name, eng, det, regime):
    lc = ov7._load("v6" if name == "v6" else "v3dens")
    if name == "v6":
        return o6.ClosureApplyV6(lc, eng, det, regime, xform=o3.ratio_transform)
    return o3.ClosureApplyV3(lc, eng, det, regime, xform=o3.ratio_transform)


def vendored_identity():
    say("1. Vendored engine")
    subs = {"exact_tc.py": [], "cumulant.py": [("from exact_tc import", "from .exact_tc import")],
            "scalable_engine.py": [("from cumulant import", "from .cumulant import")]}
    for f, s in subs.items():
        src = open(os.path.join(ROOT, "src/engines", f)).read()
        for a, b in s:
            src = src.replace(a, b)
        ven = open(os.path.join(ROOT, "portable_v7/pv7/engine", f)).read()
        check(f"pv7/engine/{f} == src/engines/{f} modulo relative imports", src == ven)


def router_grid():
    say("2. Router")
    ks = [0.0, 0.03, np.nextafter(0.045, 0), 0.045, np.nextafter(0.045, 1), 0.046, 0.15, 1.0]
    gs = [0.0, 0.02, np.nextafter(0.05, 0), 0.05, np.nextafter(0.05, 1), 0.051, 0.1, 1.0]
    bad = [(k, g) for k in ks for g in gs if proute(k, g) != ov7.route(k, g)]
    check(f"portable route == repository route on {len(ks) * len(gs)} points incl. exact boundaries", not bad, str(bad))
    check("route(0.045, 0.05) == 'v6' (inclusive)", proute(0.045, 0.05) == "v6")


def spec_checks():
    say("3. Feature specification")
    import json
    spec = json.load(open(os.path.join(ROOT, "portable_v7/feature_spec.json")))
    X = np.random.default_rng(1).normal(size=(4, 27)); X[:, 19] = 0.37
    Y = o3.ratio_transform(X)
    changed = sorted(int(c) for c in np.where(np.any(Y != X, axis=0))[0])
    check("ratio-transformed columns in feature_spec.json == columns ratio_transform changes",
          changed == spec["ratio_transform"]["divide_columns"], f"{changed}")
    check("spec widths 21 / 27 and column lists complete",
          [c["col"] for c in spec["columns_v4"]] == list(range(21)) and
          [c["col"] for c in spec["columns_v6"]] == list(range(27)))
    check("spec ratio flags agree with divide_columns",
          sorted(c["col"] for c in spec["columns_v6"] if c["ratio"]) == spec["ratio_transform"]["divide_columns"])


def feature_and_closure(eng, det, regime, name, states, hs):
    """Features (vs training-time extract_v3/extract_v6) and kappa3 (vs the original applier) at given states."""
    mem = name == "v6"
    pc = PClosure(pr.branch(name), eng, det, regime, memory=mem)
    oc = orig_closure(name, eng, det, regime)
    fmax, cmax = 0.0, {"direct": 0.0, "conjugated": 0.0, "self-conjugate": 0.0}
    nkeys = {"direct": 0, "conjugated": 0, "self-conjugate": 0}
    for m, h in zip(states, hs):
        pc.current_h = h
        oc.current_h = h
        mdict = {k: m[j] for j, k in enumerate(eng.track)}
        ctx = o3.context_v3(mdict, eng.M)
        feats = pc.features(m)
        for t, (oi, X, csign) in feats.items():
            for row, i in enumerate(oi):
                k = eng.o3_keys[i]
                sig, conj = o3.canonical_sig(k)
                kk = o3.conj_key(k) if conj else k
                plan = olc.feature_plan(kk)
                if mem:
                    tr = o6.trace_features(np.array(ctx), h)
                    ref = o6.extract_v6(mdict, plan, det, ctx, regime, tr)
                else:
                    ref = o3.extract_v3(mdict, plan, det, ctx, regime)
                ref = o3.ratio_transform(np.array([ref]))[0]
                fmax = max(fmax, float(np.max(np.abs(ref - X[row]))))
        ko, kp = oc(m), pc(m)
        for i, k in enumerate(eng.o3_keys):
            cat = "self-conjugate" if k == o3.conj_key(k) else ("conjugated" if o3.canonical_sig(k)[1] else "direct")
            nkeys[cat] += 1
            cmax[cat] = max(cmax[cat], float(abs(ko[i] - kp[i])))
            if cat == "self-conjugate":
                assert kp[i].imag == 0.0, "self-conjugate key with nonzero imaginary part"
    return fmax, cmax, nkeys


def trajectory(cell):
    if len(cell) == 8 and isinstance(cell[1], int):
        cid, M, gc, ka, gp, sg, seed, br = cell
    else:
        cid, gc, ka, gp, sg, seed, _, br = cell
        M = 5
    eng_o, w_o, _ = ope.build_cell(M, gc, ka, gp, sg, seed)
    eng_p, w_p, _ = pr.build_cell(M, gc, ka, gp, sg, seed)
    assert np.array_equal(w_o, w_p), "disorder draws differ"
    det = w_o - ope.WC
    m0 = ope.m0_product(eng_o, 1.0)
    bo, to = ov7.run_v7(eng_o, m0, det, (gc, ka, gp))
    bp, tp = pr.run_v7(eng_p, pr.m0_product(eng_p, 1.0), w_p - pr.WC, (gc, ka, gp))
    _, plo = ope.run_from(eng_o, m0, None)
    _, plp = pr.run_from(eng_p, pr.m0_product(eng_p, 1.0), None)
    ts = np.linspace(0, 40, 200)
    no, zo = oobs(eng_o, to)
    npp, zp = pr.observables(eng_p, tp)
    ipo, ipp = int(np.argmax(no)), int(np.argmax(npp))
    res = dict(branch=(bo, bp), all=float(np.max(np.abs(to - tp))), n=float(np.max(np.abs(no - npp))),
               z=float(np.max(np.abs(zo - zp))), peak=abs(no[ipo] - npp[ipp]), tpk=abs(ts[ipo] - ts[ipp]),
               plain=float(np.max(np.abs(plo - plp))))
    # states for the closure/feature check: every 20th saved point of the original trajectory
    states = to[::20]
    if br == "v6":
        clo = orig_closure("v6", eng_o, det, (gc, ka, gp))
        _, _, h = o6.run_v6(eng_o, m0, clo, o6.lambdas(ka, gp))
        hs = h[::20]
    else:
        hs = [None] * len(states)
    return cid, br, eng_p, w_p - pr.WC, (gc, ka, gp), res, states, hs


def main():
    say("PORTABLE V7 EQUIVALENCE GATES (portable_v7/pv7 vs frozen pickles + src/)")
    vendored_identity()
    router_grid()
    spec_checks()

    say("4. Clip bounds and network arrays")
    for name, pkl in (("v4", "v3dens"), ("v6", "v6")):
        lc = ov7._load(pkl)
        b = pr.branch(name)
        same_types = set(lc.nets) == set(b.nets)
        bounds = all(np.array_equal(lc.range[t][0], b.nets[t].lo) and np.array_equal(lc.range[t][1], b.nets[t].hi)
                     for t in lc.nets)
        arrays = all(all(np.array_equal(W, Wp) for W, Wp in zip(lc.nets[t].coefs_, b.nets[t].W)) and
                     all(np.array_equal(c, cp) for c, cp in zip(lc.nets[t].intercepts_, b.nets[t].b)) and
                     np.array_equal(lc.scalers[t].mean_, b.nets[t].mean) and
                     np.array_equal(lc.scalers[t].scale_, b.nets[t].scale) for t in lc.nets)
        check(f"{name}: same {len(lc.nets)} types; clip bounds bit-identical; weights/biases/scaler bit-identical",
              same_types and bounds and arrays)

    say("5-7. Rollouts, features, kappa3 and static predictions on Result 40 cells (not the external panel)")
    feat_states = {"v4": [], "v6": []}
    for cell in FIXTURE + CELLS:
        cid, br, eng, det, regime, r, states, hs = trajectory(cell)
        check(f"{cid} ({br}) branch identical", r["branch"][0] == r["branch"][1] == br, str(r["branch"]))
        check(f"{cid} trajectory: all tracked moments max|diff| {r['all']:.2e}; n {r['n']:.2e}; mean z {r['z']:.2e}; "
              f"peak {r['peak']:.2e}; t_peak {r['tpk']:.2e}; plain {r['plain']:.2e}",
              max(r["all"], r["n"], r["z"], r["peak"], r["plain"]) <= T_TRAJ and r["tpk"] == 0.0)
        fmax, cmax, nk = feature_and_closure(eng, det, regime, br, states, hs)
        check(f"{cid} features vs training-time extract_{'v6' if br == 'v6' else 'v3'}: max|diff| {fmax:.2e}",
              fmax <= T_FEAT)
        check(f"{cid} kappa3 vs original applier: " + ", ".join(f"{k} {v:.2e} ({nk[k]} key-evals)" for k, v in cmax.items()),
              max(cmax.values()) <= T_CLOS)
        pc = PClosure(pr.branch(br), eng, det, regime, memory=(br == "v6"))
        for m, h in zip(states, hs):
            pc.current_h = h
            for t, (_, X, _) in pc.features(m).items():
                feat_states[br].append((t, X))

    say("8. Static network equivalence (sklearn predict vs NumPy forward, before clipping)")
    rng = np.random.default_rng(7)
    for name, pkl in (("v4", "v3dens"), ("v6", "v6")):
        lc = ov7._load(pkl)
        b = pr.branch(name)
        worst_abs, worst_rel, n = 0.0, 0.0, 0
        per_type = {}
        for t in lc.nets:
            sc = lc.scalers[t]
            Xr = rng.normal(size=(2000, sc.mean_.size)) * 3 * sc.scale_ + sc.mean_
            Xt = [X for tt, X in feat_states[name] if tt == t]
            X = np.vstack([Xr] + Xt) if Xt else Xr
            yo = lc.nets[t].predict(sc.transform(X))
            yp = b.nets[t].predict_raw(X)
            a = float(np.max(np.abs(yo - yp)))
            rel = float(np.max(np.abs(yo - yp) / np.maximum(np.abs(yo), 1e-300)))
            per_type[t] = a
            worst_abs, worst_rel, n = max(worst_abs, a), max(worst_rel, rel), n + len(X)
        check(f"{name}: {len(lc.nets)} networks, {n} inputs (random + real trajectory features): "
              f"max abs diff {worst_abs:.2e}, max rel diff {worst_rel:.2e}; types with nonzero diff: "
              f"{sum(v > 0 for v in per_type.values())}", worst_abs <= T_STATIC)

    say(f"\nRESULT: {'ALL PASS' if not FAIL else 'FAIL: ' + ', '.join(FAIL)}")
    with open(os.path.join(ROOT, "results/portable_v7_equivalence.log"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
