"""
Held-out-parameter burst-peak evaluation at M = 5 (PROTOCOL.md, "Publication finalization P3", pre-registered
in commit bf5e15c before any run). Frozen V7, plain second order, exact mesolve; 30 cells off the training grid.

Guards (fail loudly, before any solve):
  - frozen model sha256 must match the pre-registered values;
  - every cell's support class is recomputed with training_support.classify and must match the frozen table;
  - the seed block must be disjoint from every seed previously used.

    python run.py src/evaluation/heldout_parameter_burst.py            # the campaign
    python run.py src/evaluation/heldout_parameter_burst.py --smoke    # infrastructure-only check (M=2 CONTROL,
                                                                         draw 50; not a campaign cell)
"""
import sys
import json
import time
import hashlib
import numpy as np

from pilot_eval import build_cell, m0_product, run_from, WC
from burst_pred import exact_ref_big, burst_obs
from local_run import observables, stable_ok
from v7 import run_v7, route
from training_support import classify

M = 5
NC_REF, NC_REF_FALLBACK, FOCK_GUARD = 16, 24, 0.45
CELLS_JSON = "results/heldout_param_cells.json"
OUT_LOG, OUT_NPZ = "results/heldout_param_burst.log", "results/heldout_param_burst.npz"
PRIOR_SEEDS = (set(range(0, 6)) | set(range(7, 15)) | set(range(16, 30)) | set(range(31, 43))
               | {50, 51, 52} | set(range(60, 65)) | {5, 15, 30, 105, 115, 130, 205, 215, 230, 12345})
TOL_PHYS = 1e-6


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def physicality(eng, traj, ts, tmax):
    """Per-emitter |<sz_i>| <= 1 and <a'a> >= 0 over t <= tmax. Returns (ok, worst_sz_excess, min_n)."""
    tp = eng.tpos
    nkey = (1, 1, tuple([0] * eng.M))
    zcols = [tp[(0, 0, tuple(3 if s == i else 0 for s in range(eng.M)))] for i in range(eng.M)]
    sel = ts <= tmax + 1e-12
    sz = np.array([[row[c].real for c in zcols] for row in traj[sel]])
    n = np.array([row[tp[nkey]].real for row in traj[sel]])
    excess = float(np.max(np.abs(sz)) - 1.0) if sz.size else np.nan
    nmin = float(np.min(n)) if n.size else np.nan
    return bool(excess <= TOL_PHYS and nmin >= -TOL_PHYS), excess, nmin


def run_one(Mx, gc, ka, gp, sg, seed):
    eng, w, g = build_cell(Mx, gc, ka, gp, sg, seed)
    t0 = time.time()
    ts, ne, ze, frac = exact_ref_big(Mx, w, g, ka, gp, NC_REF)
    nc_used = NC_REF
    if frac >= FOCK_GUARD:                       # pre-registered rule: rerun the same cell at Nc = 24
        ts, ne, ze, frac = exact_ref_big(Mx, w, g, ka, gp, NC_REF_FALLBACK)
        nc_used = NC_REF_FALLBACK
    t_exact = time.time() - t0
    ref = burst_obs(ts, ne, ka)
    m0 = m0_product(eng, 1.0)
    out = dict(ref=ref, nc_ref=nc_used, fock_frac=frac, t_exact=t_exact)
    tp_win = ref["tpeak"] + 2.0
    t0 = time.time()
    _, tp = run_from(eng, m0, None)
    out["t_plain"] = time.time() - t0
    t0 = time.time()
    branch, t7 = run_v7(eng, m0, np.asarray(w) - WC, (gc, ka, gp))
    out["t_v7"] = time.time() - t0
    out["branch"] = branch
    for tag, tr in (("plain", tp), ("v7", t7)):
        rec = dict(stable=bool(stable_ok(tr)))
        if rec["stable"]:
            n, _ = observables(eng, tr)
            o = burst_obs(ts, n, ka)
            rec.update(npeak=o["npeak"], tpeak=o["tpeak"],
                       signed=(o["npeak"] - ref["npeak"]) / ref["npeak"],
                       dt=o["tpeak"] - ref["tpeak"])
            rec["abs"] = abs(rec["signed"])
        else:
            rec.update(npeak=np.nan, tpeak=np.nan, signed=np.nan, abs=np.nan, dt=np.nan)
        okb, exb, nmb = physicality(eng, tr, ts, tp_win) if np.isfinite(tr).all() else (False, np.nan, np.nan)
        okf, exf, nmf = physicality(eng, tr, ts, 40.0) if np.isfinite(tr).all() else (False, np.nan, np.nan)
        rec.update(phys_burst=okb, phys_full=okf, sz_excess_burst=exb, nmin_burst=nmb,
                   sz_excess_full=exf, nmin_full=nmf)
        out[tag] = rec
    return out


def summarize(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)])
    if v.size == 0:
        return dict(n=0, median=np.nan, mean=np.nan, q25=np.nan, q75=np.nan)
    return dict(n=int(v.size), median=float(np.median(v)), mean=float(np.mean(v)),
                q25=float(np.percentile(v, 25)), q75=float(np.percentile(v, 75)))


def boot_median(v, B=10_000, seed=0):
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    meds = np.array([np.nanmedian(rng.choice(v, v.size, replace=True)) for _ in range(B)])
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def group_stats(rows):
    win = [r["win"] for r in rows]
    a7 = [r["v7"]["abs"] for r in rows]
    ap = [r["plain"]["abs"] for r in rows]
    imp = [r["plain"]["abs"] - r["v7"]["abs"] if np.isfinite(r["v7"]["abs"]) and np.isfinite(r["plain"]["abs"])
           else np.nan for r in rows]
    st = dict(n=len(rows), win_frac=float(np.mean(win)) if rows else np.nan,
              v7_unstable=int(sum(not r["v7"]["stable"] for r in rows)),
              plain_unstable=int(sum(not r["plain"]["stable"] for r in rows)),
              v7_nonphys_burst=int(sum(not r["v7"]["phys_burst"] for r in rows)),
              v7_abs=summarize(a7), plain_abs=summarize(ap), improvement=summarize(imp),
              v7_lt10=float(np.mean([np.isfinite(x) and x < 0.10 for x in a7])) if rows else np.nan)
    finite7 = [x for x in a7 if np.isfinite(x)]
    finimp = [x for x in imp if np.isfinite(x)]
    st["v7_median_ci"] = boot_median(finite7) if len(finite7) > 1 else (np.nan, np.nan)
    st["improvement_median_ci"] = boot_median(finimp) if len(finimp) > 1 else (np.nan, np.nan)
    return st


def band(BC, B, C):
    """Pre-registered interpretation bands, applied to B u C (PROTOCOL P3)."""
    wf = BC["win_frac"]
    med7, medp = BC["v7_abs"]["median"], BC["plain_abs"]["median"]
    if (wf < 0.7) or not (med7 < 0.5 * medp) or (BC["v7_unstable"] >= 0.2 * BC["n"]):
        return "FAILED TRANSFER"
    strong = (wf >= 0.9 and B["v7_unstable"] == 0 and B["v7_abs"]["median"] <= 0.03
              and C["v7_abs"]["median"] <= 0.05 and BC["v7_lt10"] >= 0.8)
    return "STRONG TRANSFER" if strong else "PARTIAL TRANSFER"


def smoke():
    o = run_one(2, 0.70, 0.15, 0.02, 0.15, 50)
    ref = np.load("results/burst_pred_exact.npz", allow_pickle=True)["data"].item()[("CONTROL", 2, 50)][0]
    d = abs(o["ref"]["Fpeak"] - ref["ref"]["Fpeak"]) / ref["ref"]["Fpeak"]
    d7 = abs(0.15 * o["v7"]["npeak"] - ref["v7"]["Fpeak"]) / ref["v7"]["Fpeak"]
    print(f"SMOKE (M=2 CONTROL draw 50, infrastructure only): imports/model/exact/closures ran; "
          f"branch={o['branch']}; exact peak matches headline to {d:.1e}, V7 to {d7:.1e}; keys={sorted(o)}")
    assert d < 1e-8 and d7 < 1e-8, "smoke regression mismatch against burst_pred_exact.npz"
    print("SMOKE PASS")


def main():
    spec = json.load(open(CELLS_JSON))
    cells = spec["cells"]
    for name, h in spec["model_sha256"].items():
        got = sha256(f"results/{name}")
        assert got == h, f"frozen model hash mismatch for {name}: {got} != {h}"
    seeds = [c["seed"] for c in cells]
    assert len(set(seeds)) == len(seeds), "duplicate seeds"
    assert not (set(seeds) & PRIOR_SEEDS), f"seed overlap with prior use: {set(seeds) & PRIOR_SEEDS}"
    for c in cells:
        p = (c["g_c"], c["kappa"], c["gamma_phi"], c["sigma"])
        k = classify(p)
        assert k["cls"] == c["cls"] and not k["on_grid"], f"{c['id']}: class {k['cls']} != frozen {c['cls']}"
        assert route(c["kappa"], c["gamma_phi"]) == c["route"], f"{c['id']}: route mismatch"

    L = []
    say = lambda s="": (print(s, flush=True), L.append(s))
    say("HELD-OUT-PARAMETER BURST-PEAK EVALUATION, M = 5 (PROTOCOL P3, pre-registered bf5e15c)")
    say(f"guards passed: model hashes, {len(cells)} classifications recomputed, seeds {min(seeds)}-{max(seeds)} fresh")
    say(f"{'id':4s} {'g_c':>5s} {'kappa':>6s} {'gphi':>6s} {'sigma':>5s} {'seed':>4s} {'cls':>3s} {'br':>3s} | "
        f"{'exact n_pk':>10s} {'t_pk':>5s} | {'plain err':>9s} | {'V7 err':>8s} {'dt7':>5s} {'stab':>4s} "
        f"{'physB':>5s} {'physF':>5s} | win")
    rows = []
    t_all = time.time()
    for c in cells:
        o = run_one(M, c["g_c"], c["kappa"], c["gamma_phi"], c["sigma"], c["seed"])
        assert o["branch"] == c["route"], f"{c['id']}: V7 took branch {o['branch']}, expected {c['route']}"
        win = bool(o["v7"]["stable"] and np.isfinite(o["v7"]["abs"]) and
                   (not np.isfinite(o["plain"]["abs"]) or o["v7"]["abs"] < o["plain"]["abs"]))
        r = dict(c, **o, win=win)
        rows.append(r)
        f = lambda x: f"{x:+.4f}" if np.isfinite(x) else "  undef"
        say(f"{c['id']:4s} {c['g_c']:5.2f} {c['kappa']:6.3f} {c['gamma_phi']:6.3f} {c['sigma']:5.2f} {c['seed']:4d} "
            f"{c['cls']:>3s} {o['branch']:>3s} | {o['ref']['npeak']:10.5f} {o['ref']['tpeak']:5.2f} | "
            f"{f(o['plain']['signed']):>9s} | {f(o['v7']['signed']):>8s} "
            f"{o['v7']['dt']:+5.2f} {'ok' if o['v7']['stable'] else 'UNST':>4s} "
            f"{'ok' if o['v7']['phys_burst'] else 'NO':>5s} {'ok' if o['v7']['phys_full'] else 'NO':>5s} | "
            f"{'V7' if win else 'plain'}   ({o['t_exact']:.0f}s exact, Nc={o['nc_ref']}, fock {o['fock_frac']:.2f})")

    groups = {"B": [r for r in rows if r["cls"] == "B"], "C": [r for r in rows if r["cls"] == "C"],
              "D": [r for r in rows if r["cls"] == "D"]}
    groups["B+C"] = groups["B"] + groups["C"]
    groups["all"] = rows
    groups["V4 (B+C)"] = [r for r in groups["B+C"] if r["branch"] == "v4"]
    groups["V6 (B+C)"] = [r for r in groups["B+C"] if r["branch"] == "v6"]
    groups["V4 (all)"] = [r for r in rows if r["branch"] == "v4"]
    groups["V6 (all)"] = [r for r in rows if r["branch"] == "v6"]
    groups["near-boundary"] = [r for r in rows if r["near_router_boundary"]]
    stats = {k: group_stats(v) for k, v in groups.items()}
    say("\nSUMMARY (|err| = absolute relative burst-peak error; improvement = |err_plain| - |err_V7|)")
    for k, s in stats.items():
        say(f"  {k:14s} n={s['n']:2d}  win {s['win_frac']:.2f}  V7 med {s['v7_abs']['median']:.4f} "
            f"[IQR {s['v7_abs']['q25']:.4f}-{s['v7_abs']['q75']:.4f}] mean {s['v7_abs']['mean']:.4f} "
            f"(boot95 {s['v7_median_ci'][0]:.4f}-{s['v7_median_ci'][1]:.4f})  | plain med {s['plain_abs']['median']:.4f} "
            f"mean {s['plain_abs']['mean']:.4f}  | impr med {s['improvement']['median']:.4f} mean "
            f"{s['improvement']['mean']:.4f} (boot95 {s['improvement_median_ci'][0]:.4f}-"
            f"{s['improvement_median_ci'][1]:.4f})  | V7 unstable {s['v7_unstable']}, plain unstable "
            f"{s['plain_unstable']}, V7 non-physical in burst window {s['v7_nonphys_burst']}, V7<10% {s['v7_lt10']:.2f}")
    verdict = band(stats["B+C"], stats["B"], stats["C"])
    say(f"\nPRE-REGISTERED BAND (B u C): {verdict}")
    fails = [r for r in rows if (not r["win"]) or (not r["v7"]["stable"]) or (not r["v7"]["phys_burst"])
             or (np.isfinite(r["v7"]["abs"]) and r["v7"]["abs"] > 0.10) or not np.isfinite(r["v7"]["abs"])]
    say(f"\nQ4 FAILURE LIST ({len(fails)} cells: V7 loses, unstable, non-physical in burst window, or |err| > 10%)")
    for r in fails:
        say(f"  {r['id']} (g_c {r['g_c']}, kappa {r['kappa']}, gphi {r['gamma_phi']}, sigma {r['sigma']}, "
            f"{r['cls']}, {r['branch']}): V7 {r['v7']['signed']:+.4f} plain {r['plain']['signed']:+.4f} "
            f"stable {r['v7']['stable']} physB {r['v7']['phys_burst']} physF {r['v7']['phys_full']}")
    say(f"\nwall time {time.time() - t_all:.0f}s")
    np.savez_compressed(OUT_NPZ, rows=np.array(rows, dtype=object), stats=np.array(stats, dtype=object),
                        verdict=np.array(verdict), spec=np.array(spec, dtype=object))
    with open(OUT_LOG, "w") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        smoke()
    else:
        main()
