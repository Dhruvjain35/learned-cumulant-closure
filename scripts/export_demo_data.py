"""Export full burst trajectories for the 30 held-out-parameter cells (Result 40) for the web demo.

Re-runs every cell exactly as src/evaluation/heldout_parameter_burst.py does (same cell construction, same exact
reference and Fock guard, same second-order and V7 rollouts), keeps the photon-number and mean-inversion
trajectories, and checks every burst-peak error against the committed record in results/heldout_param_burst.npz.
It reads the frozen models and writes only demo/data/heldout.json.

    python run.py scripts/export_demo_data.py [--workers 8]

About 6 minutes on 8 cores (the exact solves dominate).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys
import json
import time
import multiprocessing as mp

import numpy as np

from pilot_eval import build_cell, m0_product, run_from, WC
from burst_pred import exact_ref_big, burst_obs
from local_run import observables, stable_ok
from v7 import run_v7

M = 5
NC_REF, NC_REF_FALLBACK, FOCK_GUARD = 16, 24, 0.45
OUT = "demo/data/heldout.json"


def one(c):
    gc, ka, gp, sg, seed = c["g_c"], c["kappa"], c["gamma_phi"], c["sigma"], c["seed"]
    eng, w, g = build_cell(M, gc, ka, gp, sg, seed)
    t0 = time.time()
    ts, ne, ze, frac = exact_ref_big(M, w, g, ka, gp, NC_REF)
    if frac >= FOCK_GUARD:
        ts, ne, ze, frac = exact_ref_big(M, w, g, ka, gp, NC_REF_FALLBACK)
    t_exact = time.time() - t0
    ref = burst_obs(ts, ne, ka)
    m0 = m0_product(eng, 1.0)
    t0 = time.time()
    _, tp = run_from(eng, m0, None)
    t_plain = time.time() - t0
    t0 = time.time()
    branch, t7 = run_v7(eng, m0, np.asarray(w) - WC, (gc, ka, gp))
    t_v7 = time.time() - t0
    out = dict(id=c["id"], g_c=gc, kappa=ka, gamma_phi=gp, sigma=sg, seed=seed, cls=c["cls"], branch=branch,
               detunings=[float(x) for x in np.asarray(w) - WC], t=[float(x) for x in ts],
               exact=dict(n=[float(x) for x in np.real(ne)], z=[float(x) for x in np.real(ze)],
                          npeak=ref["npeak"], tpeak=ref["tpeak"], seconds=t_exact),
               timing=dict(exact=t_exact, plain=t_plain, v7=t_v7))
    for tag, tr in (("plain", tp), ("v7", t7)):
        assert stable_ok(tr), f"{c['id']} {tag} unstable"
        n, z = observables(eng, tr)
        o = burst_obs(ts, n, ka)
        out[tag] = dict(n=[float(x) for x in n], z=[float(x) for x in z], npeak=o["npeak"], tpeak=o["tpeak"],
                        err=(o["npeak"] - ref["npeak"]) / ref["npeak"])
    print(f"{c['id']}  plain {out['plain']['err']:+.4f}  v7 {out['v7']['err']:+.4f}  "
          f"({t_exact:.0f}s exact, {t_v7:.1f}s V7)", flush=True)
    return out


def main():
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 8
    cells = json.load(open("results/heldout_param_cells.json"))["cells"]
    t0 = time.time()
    with mp.get_context("fork").Pool(workers) as pool:
        rows = pool.map(one, cells, chunksize=1)

    # Every exported burst-peak error must match the committed Result 40 record.
    rec = {r["id"]: r for r in np.load("results/heldout_param_burst.npz", allow_pickle=True)["rows"]}
    worst = 0.0
    for r in rows:
        for tag in ("plain", "v7"):
            d = abs(r[tag]["err"] - rec[r["id"]][tag]["signed"])
            worst = max(worst, d)
    print(f"max |err - committed| over 60 values: {worst:.2e}")
    assert worst < 1e-9, "exported errors do not match the committed record"

    def rnd(v, k=5):
        return [float(f"{x:.{k}g}") for x in v]

    for r in rows:
        r["t"] = rnd(r["t"], 6)
        for tag in ("exact", "plain", "v7"):
            r[tag]["n"] = rnd(r[tag]["n"])
            r[tag]["z"] = rnd(r[tag]["z"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(dict(source="Result 40 (pre-registered held-out parameters, M = 5), re-run by "
                              "scripts/export_demo_data.py from the frozen models; errors match "
                              "results/heldout_param_burst.npz", M=M, cells=rows), fh, separators=(",", ":"))
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e3:.0f} kB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
