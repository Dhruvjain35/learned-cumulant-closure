"""
Interpretability steps 5+6 (PROTOCOL, locked rule): distill nested 5/10/20-term formulas per branch,
refit weights by least squares on the balanced training rows, and select in CLOSED LOOP on dev seeds
35/36 -- smallest rung with 100% branch stability AND >= 70% of V7's error reduction vs plain. Branches
never pooled: branch-N (Q21 features, ranked by normal-region contribution) evaluated on normal+seam+far-M;
branch-B (Q27 features, ranked by box contribution) evaluated on the memory-box cells. One new
instability anywhere in a branch fails that rung. With --seeds 37,38,39: the frozen-formula FINAL test.
"""
import sys
import pickle
import numpy as np

from closure_v3 import ClosureApplyV3, ratio_transform
from closure_v6 import ClosureApplyV6, run_v6, lambdas
from baseline_shell import LocalClosureShell, FormulaNet  # noqa: F401
from local_run import observables, stable_ok
from pilot_eval import build_cell, m0_product, run_from, exact_ref, nrmse, WC, NC
from campaign_eval import V0REG
from mc_ref import mc_reference
from scalable_engine import ScalableEngine
from exact_tc import draw_disorder
from interp_contrib import NAMES21, NAMES27, DOM

SEEDS = [35, 36]
FINAL = "--seeds" in sys.argv
if FINAL:
    SEEDS = [int(s) for s in sys.argv[sys.argv.index("--seeds") + 1].split(",")]
GC, SG, M5 = 0.7, 0.15, 5
TYPES = [("sp", "sm", "sz"), ("a", "sp", "sz")]
RUNGS = [5, 10, 20]
SHARE_REQ = 0.70
BRANCHES = {
    "N": dict(cols=21, names=NAMES21, qpkl="q21", region="normal",
              cells=[("normal", 0.15, 0.02), ("seam", 0.05, 0.02), ("seam", 0.05, 0.05)], farM=True,
              v7src="v4"),
    "B": dict(cols=27, names=NAMES27, qpkl="q27", region="box",
              cells=[("box", 0.03, 0.02), ("box", 0.045, 0.02), ("box", 0.03, 0.05),
                     ("box", 0.045, 0.05)], farM=False, v7src="v6"),
}


def build_formula_lc(branch, rung):
    """Distill: pool = contribution-ranked (branch region) x stability(freq>=0.8, sign>=0.9); take top
    `rung` terms per dominant type-group; least-squares refit on the balanced rows; wrap as an lc."""
    cfg = BRANCHES[branch]
    contrib = np.load("results/interp_contrib.npz", allow_pickle=True)["tables"].item()
    stab = np.load("results/interp_stability.npz", allow_pickle=True)["results"].item()
    pool_data = np.load("results/train_v6.npz", allow_pickle=True)["data"].item()
    with open(f"results/lc_{cfg['qpkl']}.pkl", "rb") as fh:
        qlc = pickle.load(fh)
    import fit_v56
    fit_v56.RNG = np.random.default_rng(0)
    lc = LocalClosureShell()
    lc.types = TYPES
    picked = {}
    for typ in TYPES:
        ctab = contrib[(cfg["region"], cfg["qpkl"])] if typ == DOM else None
        stable_terms = {t for t, (f, s) in stab[(typ, cfg["qpkl"])].items() if f >= 0.8 and s >= 0.9}
        qnet = qlc.nets[typ]
        poly = qnet.named_steps["poly"]
        tnames = list(poly.get_feature_names_out(cfg["names"]))
        if ctab is not None:
            ranked = [t for t, _ in sorted(ctab.items(), key=lambda kv: -kv[1][0])]
        else:  # non-DOM group: rank by production |coef| magnitude within stable set (documented fallback)
            W = qnet.named_steps["lasso"].coef_
            mag = np.max(np.abs(W), axis=0)
            ranked = [tnames[j] for j in np.argsort(-mag)]
        chosen = [t for t in ranked if t in stable_terms][:rung]
        sel_idx = [tnames.index(t) for t in chosen]
        # least-squares refit on balanced rows (same rows as every model: replay fit_v56 indices)
        X, Y, P = pool_data[typ]
        bidx = fit_v56.balanced_indices(P, np.load("results/train_v6.npz", allow_pickle=True)["points"],
                                        min(len(X), fit_v56.CAP))
        Z = qlc.scalers[typ].transform(ratio_transform(X[bidx][:, :cfg["cols"]]))
        Phi = poly.transform(Z)[:, sel_idx]
        A = np.hstack([Phi, np.ones((len(Phi), 1))])
        sol, *_ = np.linalg.lstsq(A, Y[bidx], rcond=None)
        lc.scalers[typ] = qlc.scalers[typ]
        lc.nets[typ] = FormulaNet(poly, sel_idx, sol[:-1], sol[-1], chosen)
        lc.range[typ] = qlc.range[typ]
        picked[typ] = chosen
    return lc, picked


def branch_eval(branch, lc):
    cfg = BRANCHES[branch]
    rows = []
    for (rt, ka, gp) in cfg["cells"]:
        for sd in SEEDS:
            eng, w, g = build_cell(M5, GC, ka, gp, SG, sd)
            det = w - WC
            ne, ze, _ = exact_ref(M5, w, g, ka, gp)
            m0 = m0_product(eng, 1.0)
            def roll(model_lc, traced):
                if traced:
                    _, tr, _ = run_v6(eng, m0, ClosureApplyV6(model_lc, eng, det, (GC, ka, gp),
                                                              xform=ratio_transform), lambdas(ka, gp))
                else:
                    _, tr = run_from(eng, m0, ClosureApplyV3(model_lc, eng, det, (GC, ka, gp),
                                                             xform=ratio_transform))
                return tr
            _, tp = run_from(eng, m0, None)
            plain = nrmse(observables(eng, tp)[0], ne) if stable_ok(tp) else np.inf
            with open(f"results/lc_{'v3dens' if cfg['v7src']=='v4' else 'v6'}.pkl", "rb") as fh:
                v7lc = pickle.load(fh)
            t7 = roll(v7lc, cfg["v7src"] == "v6")
            v7 = nrmse(observables(eng, t7)[0], ne) if stable_ok(t7) else np.inf
            tf = roll(lc, branch == "B")
            okf = stable_ok(tf)
            fv = nrmse(observables(eng, tf)[0], ne) if okf else np.inf
            share = (plain - fv) / (plain - v7) if np.isfinite(v7) and plain > v7 else np.nan
            rows.append((okf, fv, share))
            print(f"    [{rt}] ka={ka:.3f} gp={gp:.3f} s{sd}: formula={'X' if not okf else f'{fv:.3f}'} "
                  f"plain={plain:.3f} V7={v7:.3f} share={share:.2f}", flush=True)
    if cfg["farM"]:
        gc0, ka0, gp0, sg0 = V0REG
        for M in (9, 10):
            for sd in SEEDS[:2]:
                w = draw_disorder(M, WC, sg0, sd, "gauss")
                g = np.full(M, gc0 / np.sqrt(M))
                eng = ScalableEngine(M, NC, WC, w, g, kappa=ka0, gdown=0.02, gphi=gp0, Fsafe=6).compile()
                det = w - WC
                _, nref, zref, mcerr = mc_reference(M, NC, w, g, ntraj=200)
                m0 = m0_product(eng, 1.0)
                _, tf = run_from(eng, m0, ClosureApplyV3(lc, eng, det, (gc0, ka0, gp0),
                                                         xform=ratio_transform))
                okf = stable_ok(tf)
                fv = nrmse(observables(eng, tf)[0], nref) if okf else np.inf
                rows.append((okf, fv, np.nan))
                print(f"    [farM] M={M} s{sd}: formula={'X' if not okf else f'{fv:.3f}'} "
                      f"(MCerr {mcerr:.3f})", flush=True)
    stabr = np.mean([r[0] for r in rows])
    med_share = float(np.nanmedian([r[2] for r in rows]))
    return stabr, med_share


if __name__ == "__main__":
    verdicts = {}
    for branch in ("N", "B"):
        print(f"\nBRANCH {branch} ({BRANCHES[branch]['qpkl']} features, "
              f"{'final' if FINAL else 'selection'} seeds {SEEDS}):")
        chosen_rung = None
        for rung in (RUNGS if not FINAL else [np.load('results/formula_choice.npz',
                                                      allow_pickle=True)['choice'].item()[branch]]):
            if rung is None:
                print(f"  branch {branch}: no rung was selected in dev -> no final test")
                break
            lc, picked = build_formula_lc(branch, rung)
            print(f"  rung {rung}: terms {picked[DOM][:min(rung,8)]}{'...' if rung > 8 else ''}")
            stabr, med_share = branch_eval(branch, lc)
            print(f"  rung {rung}: stability {100*stabr:.0f}%, median advantage share {med_share:.2f}")
            if stabr == 1.0 and med_share >= SHARE_REQ:
                chosen_rung = rung
                verdicts[branch] = (rung, stabr, med_share, picked)
                print(f"  -> rung {rung} {'PASSES FINAL' if FINAL else 'SELECTED (smallest qualifying)'}")
                break
            elif FINAL:
                verdicts[branch] = (rung, stabr, med_share, picked)
                print(f"  -> FINAL FAIL (stability {100*stabr:.0f}%, share {med_share:.2f})")
        if not FINAL and chosen_rung is None:
            verdicts[branch] = (None, None, None, None)
            print(f"  -> NO compact formula qualifies for branch {branch}: V7 stands as a validated "
                  f"computational closure there")
    if not FINAL:
        np.savez_compressed("results/formula_choice.npz",
                            choice=np.array({b: v[0] for b, v in verdicts.items()}, dtype=object))
    np.savez_compressed(f"results/interp_formula_{'final' if FINAL else 'dev'}.npz",
                        verdicts=np.array(verdicts, dtype=object))
    print(f"\nsaved formula {'final' if FINAL else 'selection'} verdicts")
