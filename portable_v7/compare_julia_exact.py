"""
OPTIONAL Level-2 step: compare the run's Python/QuTiP exact reference with julia/exact_panel.jl's QuantumOptics
exact reference, and recompute both closures' errors against the Julia reference.

    python compare_julia_exact.py [external_output]
Writes external_output/julia_comparison.json and prints a table.
"""
import os
import sys
import csv
import json

d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "external_output")
res = json.load(open(os.path.join(d, "results.json")))["results"]
jl = {r["cell"]: r for r in csv.DictReader(open(os.path.join(d, "julia_exact.csv")))}
rows = []
for r in res:
    j = jl[r["panel_id"]]
    nj = float(j["n_peak_exact"])
    row = dict(cell=r["panel_id"], exact_python=r["exact_npeak"], exact_julia=nj,
               exact_rel_diff=(r["exact_npeak"] - nj) / nj,
               plain_err_vs_julia=None if r["plain_npeak"] is None else (r["plain_npeak"] - nj) / nj,
               v7_err_vs_julia=None if r["v7_npeak"] is None else (r["v7_npeak"] - nj) / nj)
    rows.append(row)
    print(f"{row['cell']}: exact python {row['exact_python']:.6f} julia {nj:.6f} (rel diff {row['exact_rel_diff']:+.1e}) | "
          f"plain vs julia {row['plain_err_vs_julia']:+.4f} | V7 vs julia {row['v7_err_vs_julia']:+.4f}")
json.dump(rows, open(os.path.join(d, "julia_comparison.json"), "w"), indent=1)
