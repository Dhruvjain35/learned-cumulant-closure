"""Export the demo's showcase cell and emitter-count ladder from committed results (no simulation).

    python run.py scripts/export_demo_static.py

Writes demo/data/showcase.json (results/baseline_burst_trace_M3_pocket_draw50.npz: exact, 2nd, 3rd, 4th order and
the learned closure on one cell) and demo/data/scaling.json (the Result 34-37 ladder logs).
"""
import os
import json
import re
import numpy as np

R, OUT = "results/", "demo/data/"


def rnd(v, k=5):
    return [float(f"{x:.{k}g}") for x in v]


def showcase():
    d = np.load(R + "baseline_burst_trace_M3_pocket_draw50.npz", allow_pickle=True)
    meta = d["meta"].item()
    ex = d["n_exact"]
    pk = float(ex.max())
    out = dict(M=meta["M"], g_c=meta["gc"], kappa=meta["kappa"], gamma_phi=meta["gphi"], sigma=meta["sigma"],
               seed=meta["seed"], branch=meta["v7_branch"], methods={})
    for k in ("exact", "v7", "plain", "t3", "t4"):
        t, n = d["ts_" + k], d["n_" + k]
        early = t < 6
        npk = float(n[early].max())
        out["methods"][k] = dict(t=rnd(t, 6), n=rnd(n), npeak=npk, err=(npk - pk) / pk,
                                 diverged=bool(k == "t4" and meta["t4_guard_triggered"]),
                                 stop=float(t[-1]))
    return out


def scaling():
    pat = re.compile(r"^(CONTROL|POCKET)\s+(\d+)\s+(?:v\d\s+)?\|\s+([\d.]+)\s+(?:([\d.]+)%\s+\|\s+)?([+-][\d.]+)\s+([+-][\d.]+)")
    rows = []
    for name, ref in (("v8_accuracy_ladder.log", "exact"), ("v8_accuracy_M16.log", "exact"),
                      ("v8_disordered_ladder.log", "trajectories"), ("v8_disordered_m12.log", "trajectories")):
        for line in open(R + name):
            m = pat.match(line.strip())
            if m:
                rows.append(dict(regime=m[1], M=int(m[2]), reference=ref, disorder=ref == "trajectories",
                                 npeak=float(m[3]), mc_err_pct=float(m[4]) if m[4] else None,
                                 v7=float(m[5]), plain=float(m[6]), source=f"results/{name}"))
    return rows


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    json.dump(showcase(), open(OUT + "showcase.json", "w"), separators=(",", ":"))
    json.dump(scaling(), open(OUT + "scaling.json", "w"), separators=(",", ":"))
    print("wrote demo/data/showcase.json, demo/data/scaling.json")
