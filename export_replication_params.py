"""
Export the EXACT model parameters (including the disorder draws w_i) for the burst-vs-M cells to JSON, so an
independent QuantumCumulants.jl / QuantumOptics.jl reproduction uses identical detunings and couplings -- an
exact implementation cross-check, not just a qualitative one. Parameters only (no results) -> no leakage.
"""
import json
import numpy as np
from exact_tc import draw_disorder

WC, GDOWN, SG = 1.0, 0.02, 0.15
REGIMES = {"POCKET": (0.70, 0.03, 0.02), "CONTROL": (0.70, 0.15, 0.02)}
MS = [2, 3, 4, 5]
SEEDS = [50, 51, 52]

cells = []
for name, (gc, ka, gp) in REGIMES.items():
    for M in MS:
        for sd in SEEDS:
            w = draw_disorder(M, WC, SG, sd, "gauss")           # per-emitter frequencies (disorder)
            g = np.full(M, gc / np.sqrt(M))                     # per-emitter coupling g_i = g_c/sqrt(M)
            cells.append(dict(regime=name, M=int(M), seed=int(sd), gc=gc, kappa=ka, gdown=GDOWN, gphi=gp,
                              wc=WC, sigma=SG, w=[float(x) for x in w], g=[float(x) for x in g]))

meta = dict(model="open disordered Tavis-Cummings (RWA)",
            hamiltonian="H = wc*a'a + sum_i w_i*sp_i*sm_i + sum_i g_i*(a*sp_i + a'*sm_i)",
            dissipators="kappa*D[a] + gdown*D[sm_i] + (gphi/2)*D[sz_i]  (per emitter i)",
            initial_state="cavity vacuum; every emitter EXCITED (fully inverted, superradiant IC)",
            observable="emitted flux kappa*<a'a>(t); compare F_peak=kappa*max_t<a'a>, t_peak=argmax_t<a'a>",
            convention_note="sz = |e><e| - |g><g|; sp=|e><g|; sm=|g><e|. tlist t in [0,40], 200 points.")
with open("results/replication_params.json", "w") as fh:
    json.dump(dict(meta=meta, cells=cells), fh, indent=1)
print(f"wrote results/replication_params.json: {len(cells)} cells "
      f"({len(REGIMES)} regimes x {len(MS)} M x {len(SEEDS)} seeds)")
