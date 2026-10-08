"""
External reproduction runner for the frozen V7 closure (docs/EXTERNAL_V7_REPRODUCTION_PROTOCOL.md).

From this directory:   python run_external_v7_reproduction.py
  1. runs the self-check (run_selfcheck.py) and stops if it fails;
  2. for every cell of external_panel.json: draws the disorder from the cell's seed, runs the exact reference
     (QuTiP mesolve, Fock cutoff 16), plain second order, and the portable frozen V7; records burst peaks, signed
     and absolute relative errors, the V7 branch, stability, and physicality;
  3. writes external_output/: results.json, results.csv, run.log, environment.json, and
     panel_params_for_julia.json (the same cells, with the drawn disorder, for julia/exact_panel.jl).
Nothing here is tunable: no retraining, no router or clipping change, no cell or seed change.

Options (for testing the installation only; the protocol run uses neither):
  --panel FILE   use another panel file
  --out DIR      write somewhere other than external_output/
"""
import os
import sys
import csv
import json
import time
import platform
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def arg(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def main():
    panel_path = arg("--panel", os.path.join(HERE, "external_panel.json"))
    out_dir = arg("--out", os.path.join(HERE, "external_output"))
    os.makedirs(out_dir, exist_ok=True)
    log = []

    def say(s=""):
        print(s, flush=True)
        log.append(s)

    from run_selfcheck import run_selfcheck
    say("STEP 1: self-check (SELF-CHECK FIXTURE -- NOT AN INDEPENDENT RESULT)")
    ok, env = run_selfcheck(verbose=True)
    if not ok:
        say("self-check FAILED: stopping before any panel cell. See selfcheck_output/selfcheck.json.")
        sys.exit(1)

    import pv7.rollout as pr
    from pv7.router import route
    panel = json.load(open(panel_path))
    say(f"\nSTEP 2: panel {os.path.basename(panel_path)} ({len(panel['cells'])} cells), M = {panel['M']}")
    rows, julia_cells = [], []
    t_all = time.time()
    for c in panel["cells"]:
        M, gc, ka, gp, sg, seed = panel["M"], c["g_c"], c["kappa"], c["gamma_phi"], c["sigma"], c["seed"]
        t0 = time.time()
        eng, w, g = pr.build_cell(M, gc, ka, gp, sg, seed)
        ts, ne, _, frac = pr.exact_ref(M, w, g, ka, gp)
        nc_used = 16
        if frac >= 0.45:
            ts, ne, _, frac = pr.exact_ref(M, w, g, ka, gp, Nc=24)
            nc_used = 24
        ref = pr.burst_obs(ts, ne, ka)
        m0 = pr.m0_product(eng, 1.0)
        _, tp = pr.run_from(eng, m0, None)
        br, t7 = pr.run_v7(eng, m0, w - pr.WC, (gc, ka, gp))
        r = dict(panel_id=c.get("panel_id", c.get("id")), parameter_cell=c.get("result40_parameter_cell", ""),
                 g_c=gc, kappa=ka, gamma_phi=gp, sigma=sg, seed=seed, support_class=c.get("support_class", ""),
                 expected_branch=c.get("branch", route(ka, gp)), branch=br, w=[float(x) for x in w],
                 exact_npeak=ref["npeak"], exact_tpeak=ref["tpeak"], nc_ref=nc_used, fock_fraction=frac)
        win = (ref["tpeak"] + 2.0)
        for tag, tr in (("plain", tp), ("v7", t7)):
            stable = bool(pr.stable_ok(tr))
            if stable:
                n, _ = pr.observables(eng, tr)
                o = pr.burst_obs(ts, n, ka)
                r[f"{tag}_npeak"], r[f"{tag}_tpeak"] = o["npeak"], o["tpeak"]
                r[f"{tag}_signed_err"] = (o["npeak"] - ref["npeak"]) / ref["npeak"]
                r[f"{tag}_abs_err"] = abs(r[f"{tag}_signed_err"])
            else:
                for k in ("npeak", "tpeak", "signed_err", "abs_err"):
                    r[f"{tag}_{k}"] = None
            finite = bool(np.isfinite(tr).all())
            r[f"{tag}_stable"] = stable
            r[f"{tag}_physical_burst_window"] = pr.physicality(eng, tr, ts, win) if finite else False
            r[f"{tag}_physical_full"] = pr.physicality(eng, tr, ts, 40.0) if finite else False
        r["v7_closer_than_plain"] = bool(r["v7_stable"] and r["v7_abs_err"] is not None and
                                         (r["plain_abs_err"] is None or r["v7_abs_err"] < r["plain_abs_err"]))
        r["seconds"] = time.time() - t0
        rows.append(r)
        julia_cells.append(dict(regime=r["panel_id"], M=M, seed=seed, gc=gc, kappa=ka, gdown=pr.GDOWN, gphi=gp,
                                wc=pr.WC, sigma=sg, w=r["w"], g=[float(x) for x in g]))
        f = lambda x: f"{x:+.4f}" if x is not None else " unstable"
        say(f"  {r['panel_id']:4s} ({r['parameter_cell']}, class {r['support_class']}, {br}) seed {seed}: "
            f"exact n_peak {ref['npeak']:.5f} | plain {f(r['plain_signed_err'])} | V7 {f(r['v7_signed_err'])} | "
            f"V7 physical {r['v7_physical_full']} | {r['seconds']:.0f}s")
    leaked = sorted(m for m in sys.modules if m.split(".")[0] in ("sklearn", "joblib"))
    import scipy
    import qutip
    env = dict(python=platform.python_version(), platform=platform.platform(), numpy=np.__version__,
               scipy=scipy.__version__, qutip=qutip.__version__, sklearn_or_joblib_imported=leaked,
               panel_file=os.path.basename(panel_path), wall_seconds=time.time() - t_all)
    say(f"\nV7 closer to exact than plain in {sum(r['v7_closer_than_plain'] for r in rows)}/{len(rows)} cells; "
        f"wall time {env['wall_seconds']:.0f}s; scikit-learn/joblib imported: {leaked or 'no'}")
    with open(os.path.join(out_dir, "results.json"), "w") as fh:
        json.dump(dict(panel=panel, results=rows), fh, indent=1)
    keys = [k for k in rows[0] if k != "w"]
    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)
    with open(os.path.join(out_dir, "environment.json"), "w") as fh:
        json.dump(env, fh, indent=1)
    with open(os.path.join(out_dir, "panel_params_for_julia.json"), "w") as fh:
        json.dump(dict(meta=dict(source="run_external_v7_reproduction.py", note="disorder drawn by this run"),
                       cells=julia_cells), fh, indent=1)
    with open(os.path.join(out_dir, "run.log"), "w") as fh:
        fh.write("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
