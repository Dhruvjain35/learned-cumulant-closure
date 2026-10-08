"""
Build portable_v7/selfcheck_expected.json: a SELF-CHECK FIXTURE, NOT AN INDEPENDENT RESULT.

Expected values are produced by the ORIGINAL frozen stack (sklearn pickles + src/ code), never by pv7, so the
external self-check compares the portable package against the originals. Cells: M = 2, draw 50 (the exported,
published replication draw; not a panel cell), one V4-routed (kappa 0.15) and one V6-routed (kappa 0.03) case.

    python run.py src/tools/make_portable_selfcheck.py
"""
import json
import warnings
import numpy as np

warnings.filterwarnings("ignore")
import pilot_eval as pe                                  # noqa: E402
import v7 as ov7                                         # noqa: E402
from burst_pred import exact_ref_big, burst_obs          # noqa: E402
from local_run import observables                        # noqa: E402

CASES = [("FIX-V4", 2, 0.70, 0.15, 0.02, 0.15, 50), ("FIX-V6", 2, 0.70, 0.03, 0.02, 0.15, 50)]


def main():
    out = dict(label="SELF-CHECK FIXTURE -- NOT AN INDEPENDENT RESULT",
               purpose="software/installation check of portable_v7 against the frozen originals; values produced by "
                       "the original sklearn-pickle stack in the source repository",
               cases=[], static=[])
    for cid, M, gc, ka, gp, sg, seed in CASES:
        eng, w, g = pe.build_cell(M, gc, ka, gp, sg, seed)
        ts, ne, _, _ = exact_ref_big(M, w, g, ka, gp, 16)
        ref = burst_obs(ts, ne, ka)
        m0 = pe.m0_product(eng, 1.0)
        _, tp = pe.run_from(eng, m0, None)
        br, t7 = ov7.run_v7(eng, m0, w - pe.WC, (gc, ka, gp))
        n7, z7 = observables(eng, t7)
        npl, _ = observables(eng, tp)
        out["cases"].append(dict(id=cid, M=M, g_c=gc, kappa=ka, gamma_phi=gp, sigma=sg, seed=seed,
                                 w=[float(x) for x in w], branch=br,
                                 exact_npeak=ref["npeak"], exact_tpeak=ref["tpeak"],
                                 plain_npeak=float(np.max(npl)), v7_npeak=float(np.max(n7)),
                                 v7_tpeak=float(ts[int(np.argmax(n7))]),
                                 v7_n_t=[float(x) for x in n7], v7_meanz_t=[float(x) for x in z7]))
    rng = np.random.default_rng(11)
    for name, pkl in (("v4", "v3dens"), ("v6", "v6")):
        lc = ov7._load(pkl)
        for t in sorted(lc.nets)[:4]:
            sc = lc.scalers[t]
            X = rng.normal(size=(4, sc.mean_.size)) * 2 * sc.scale_ + sc.mean_
            y = lc.nets[t].predict(sc.transform(X))
            out["static"].append(dict(branch=name, type=list(t), X=X.tolist(), y_raw=y.tolist()))
    with open("portable_v7/selfcheck_expected.json", "w") as fh:
        json.dump(out, fh, indent=1)
    for c in out["cases"]:
        print(c["id"], c["branch"], "exact", round(c["exact_npeak"], 6), "plain", round(c["plain_npeak"], 6),
              "v7", round(c["v7_npeak"], 6))


if __name__ == "__main__":
    main()
