"""
Prospective M = 5 confirmation of the exploratory Result 25 mechanism (PROTOCOL.md, "Publication finalization P5").

Reuses the FROZEN Result 25 machinery unchanged: mechanism.run_cell (exact kappa3, E_f, D_f, closed-loop family
ablation), mechanism.build_families (canonical_sig families), the frozen V4/V6 pickles. This file adds only:
per-seed D shares, the pre-registered decision rule, a V7 stability/physicality record, and output files.

Modes
  --regression   Result 25 reproduction: M = 4, seeds 37/38/39, both regimes, printed in mechanism.py's format,
                 compared line by line with the committed results/mechanism.log.   -> results/mechanism_m4_regression.log
  --campaign     the pre-registered M = 5 confirmation: seeds 500-505, both regimes (12 cells).
                 -> results/mechanism_m5.log, results/mechanism_m5.npz (slim), results/mechanism_m5_fig.png
T3 (kappa4 = 0) trajectories are not computed (want_t3=False): they enter only Result 25's figure, never its
decision quantities, and the T3 engine is capped at M <= 4.
"""
import sys
import time
import traceback
import numpy as np

import mechanism as mech
from mechanism import run_cell, fam_label, load
from pilot_eval import build_cell, m0_product, WC
from v7 import run_v7
from local_run import stable_ok

REGIMES = {"POCKET": (0.70, 0.03, 0.02), "CONTROL": (0.70, 0.15, 0.02)}
AAZ, APZ = ("ad", "a", "sz"), ("a", "sp", "sz")
CAMPAIGN_SEEDS = [500, 501, 502, 503, 504, 505]
TOL_PHYS = 1e-6

# pre-registered thresholds (PROTOCOL P5)
TH = dict(aaz_share_median=0.50, aaz_dominant_min=5, apz_share_median=0.15, apz_share_min=4,
          aaz_loss_median=0.50, aaz_loss_pos_min=5, apz_loss_median=0.15, apz_loss_pos_min=4)


def v7_physicality(M, gc, ka, gp, seed):
    eng, w, g = build_cell(M, gc, ka, gp, mech.SG, seed)
    br, tr = run_v7(eng, m0_product(eng, 1.0), w - WC, (gc, ka, gp))
    tp = eng.tpos
    zc = [tp[(0, 0, tuple(3 if s == i else 0 for s in range(M)))] for i in range(M)]
    n = np.array([r[tp[(1, 1, tuple([0] * M))]].real for r in tr])
    sz = np.array([[r[c].real for c in zc] for r in tr])
    ok = bool(np.isfinite(tr).all() and np.max(np.abs(sz)) <= 1 + TOL_PHYS and np.min(n) >= -TOL_PHYS)
    return bool(stable_ok(tr)), ok


def v7_repro(r, f):
    """Result 25's mechanism_report.v7_repro, per cell: ||V7 kappa3|| / ||exact kappa3|| over the burst window on
    family f's keys (teacher-forced), and the sign of their real inner product."""
    idx, win = r["fam2idx"][f], r["win"]
    ex = r["kap_ex"][np.ix_(win, idx)]
    v7 = r["kap_v7_tf"][np.ix_(win, idx)]
    ne = np.linalg.norm(ex)
    if ne <= 1e-10:
        return np.nan, np.nan
    return float(np.linalg.norm(v7) / ne), float(np.sign(np.real(np.sum(np.conj(ex) * v7))))


def run_with_retry(M, regime, gc, ka, gp, sd, v4, v6, say):
    """Pre-registered technical-failure rule: an exception is logged; the SAME cell is rerun once unchanged."""
    for attempt in (1, 2):
        try:
            return run_cell(M, regime, gc, ka, gp, sd, v4, v6, want_t3=False), attempt
        except Exception:
            say(f"    TECHNICAL FAILURE {regime} seed {sd} attempt {attempt}:\n" + traceback.format_exc())
    return None, 2


def regression():
    """Result 25 at M = 4, printed exactly as mechanism.py prints its table, then diffed against the log."""
    v4, v6 = load("v3dens"), load("v6")
    seeds = [37, 38, 39]
    out = [f"MECHANISM seeds={seeds}; cells POCKET M4, CONTROL M4; NC_ref={mech.NC_REF}", ""]
    for regime, (gc, ka, gp) in REGIMES.items():
        res = {s: run_cell(4, regime, gc, ka, gp, s, v4, v6, want_t3=False) for s in seeds}
        assert all(res[s]["gate0"] < 1e-6 for s in seeds)
        out.append(f"[{regime} M4] gate0<1e-6 OK; branch={res[seeds[0]]['br']}; "
                   f"err_plain~{np.mean([res[s]['err_plain'] for s in seeds]):.3f} "
                   f"err_v7~{np.mean([res[s]['err_v7'] for s in seeds]):.3f}")
        fams = res[seeds[0]]["fams"]
        totE = {f: np.mean([res[s]["score_E"][f] for s in seeds]) for f in fams}
        totD = {f: np.mean([res[s]["score_D"][f] for s in seeds]) for f in fams}
        SE = sum(totE.values()) or 1.0; SD = sum(totD.values()) or 1.0
        rankedD = sorted(fams, key=lambda f: -totD[f])
        out.append(f"  family  |  E_share  D_share  frac_lost(abl)  V7mag  V7sign   (ranked by D_f, top 8)")
        for f in rankedD[:8]:
            fls = [res[s]["frac_lost"].get(f, np.nan) for s in seeds]
            fl = np.nanmean([x for x in fls if np.isfinite(x)]) if any(np.isfinite(x) for x in fls) else np.nan
            ncrit = sum(np.isinf(x) for x in fls)
            rat, sgn = [], []
            for s in seeds:
                a = res[s]["Eex"][f]; b = res[s]["Ev7"][f]
                if np.mean(np.abs(a)) > 1e-12:
                    rat.append(np.mean(np.abs(b)) / np.mean(np.abs(a))); sgn.append(np.sign(np.sum(a * b)))
            crit = f" CRIT×{ncrit}" if ncrit else ""
            out.append(f"   {fam_label(f):7s} | {totE[f]/SE:6.2f}  {totD[f]/SD:6.2f}  {fl:+13.2f}  "
                       f"{np.mean(rat) if rat else float('nan'):5.2f}  {np.mean(sgn) if sgn else float('nan'):+5.1f}{crit}")
    committed = [l.rstrip() for l in open("results/mechanism.log") if l.strip() and not l.startswith("(")]
    produced = [l.rstrip() for l in out if l.strip()]
    same = committed == produced
    report = ["RESULT 25 REGRESSION (M = 4, seeds 37/38/39; current code vs committed results/mechanism.log)", ""]
    report += produced + ["", f"line-by-line identical to the committed log (all {len(committed)} lines): {same}"]
    if not same:
        for a, b in zip(committed, produced):
            if a != b:
                report.append(f"  committed: {a}\n  produced:  {b}")
    print("\n".join(report))
    with open("results/mechanism_m4_regression.log", "w") as fh:
        fh.write("\n".join(report) + "\n")
    if not same:
        sys.exit(1)


def decide(cells):
    """Pre-registered rule, applied per regime. Returns criteria dict."""
    sh_aaz = np.array([c["share"][AAZ] for c in cells])
    sh_apz = np.array([c["share"][APZ] for c in cells])
    l_aaz = np.array([c["loss"][AAZ] for c in cells])
    l_apz = np.array([c["loss"][APZ] for c in cells])
    pos = lambda x: np.array([np.isfinite(v) and v > 0 or (np.isinf(v) and v > 0) for v in x])
    C1 = (np.median(sh_aaz) > TH["aaz_share_median"]) and (np.sum(sh_aaz > sh_apz) >= TH["aaz_dominant_min"])
    C2 = (np.median(sh_apz) >= TH["apz_share_median"]) and (np.sum(sh_apz >= 0.15) >= TH["apz_share_min"])
    C3 = (np.nanmedian(l_aaz) >= TH["aaz_loss_median"]) and (np.sum(pos(l_aaz)) >= TH["aaz_loss_pos_min"])
    C4 = (np.nanmedian(l_apz) >= TH["apz_loss_median"]) and (np.sum(pos(l_apz)) >= TH["apz_loss_pos_min"])
    return dict(C1=bool(C1), C2=bool(C2), C3=bool(C3), C4=bool(C4),
                med_sh_aaz=float(np.median(sh_aaz)), med_sh_apz=float(np.median(sh_apz)),
                n_aaz_dom=int(np.sum(sh_aaz > sh_apz)), n_apz_015=int(np.sum(sh_apz >= 0.15)),
                med_l_aaz=float(np.nanmedian(l_aaz)), med_l_apz=float(np.nanmedian(l_apz)),
                n_l_aaz_pos=int(np.sum(pos(l_aaz))), n_l_apz_pos=int(np.sum(pos(l_apz))),
                iqr_sh_aaz=[float(np.percentile(sh_aaz, 25)), float(np.percentile(sh_aaz, 75))],
                iqr_sh_apz=[float(np.percentile(sh_apz, 25)), float(np.percentile(sh_apz, 75))],
                iqr_l_aaz=[float(np.nanpercentile(l_aaz, 25)), float(np.nanpercentile(l_aaz, 75))],
                iqr_l_apz=[float(np.nanpercentile(l_apz, 25)), float(np.nanpercentile(l_apz, 75))])


def campaign():
    v4, v6 = load("v3dens"), load("v6")
    L = []

    def say(s=""):
        print(s, flush=True)
        L.append(s)

    say("PROSPECTIVE M = 5 MECHANISM CONFIRMATION (PROTOCOL P5); frozen V7; seeds "
        f"{CAMPAIGN_SEEDS}; regimes {REGIMES}")
    t0 = time.time()
    cells, slim, failures = {r: [] for r in REGIMES}, [], []
    for regime, (gc, ka, gp) in REGIMES.items():
        for sd in CAMPAIGN_SEEDS:
            r, attempts = run_with_retry(5, regime, gc, ka, gp, sd, v4, v6, say)
            if r is None:
                failures.append((regime, sd))
                cells[regime].append(dict(seed=sd, share={AAZ: np.nan, APZ: np.nan},
                                          loss={AAZ: np.nan, APZ: np.nan}, failed=True))
                say(f"  {regime} seed {sd}: TECHNICAL FAILURE after rerun (counted against confirmation)")
                continue
            tot = sum(r["score_D"].values())
            share = {f: r["score_D"][f] / tot for f in r["fams"]}
            other = 1.0 - share[AAZ] - share[APZ]
            stab, phys = v7_physicality(5, gc, ka, gp, sd)
            loss = {f: (r["frac_lost"][f] if stab else np.nan) for f in (AAZ, APZ)}
            mag_aaz, sgn_aaz = v7_repro(r, AAZ)
            mag_apz, sgn_apz = v7_repro(r, APZ)
            totE = sum(r["score_E"].values()) or 1.0
            ranked = sorted(r["fams"], key=lambda f: -r["score_D"][f])
            c = dict(seed=sd, share=share, loss=loss, other=other, rank_aaz=ranked.index(AAZ) + 1,
                     rank_apz=ranked.index(APZ) + 1, E_share_aaz=r["score_E"][AAZ] / totE,
                     gate0=r["gate0"], err_v7=r["err_v7"], err_plain=r["err_plain"], branch=r["br"],
                     v7_stable=stab, v7_physical=phys, attempts=attempts,
                     abl_unstable={fam_label(f): bool(r["abl_unstable"][f]) for f in (AAZ, APZ)},
                     mag_aaz=mag_aaz, sgn_aaz=sgn_aaz, mag_apz=mag_apz, sgn_apz=sgn_apz,
                     frac_lost_all={fam_label(f): float(v) for f, v in r["frac_lost"].items()},
                     share_all={fam_label(f): float(v) for f, v in share.items()})
            cells[regime].append(c)
            win = r["win"]
            slim.append(dict(regime=regime, seed=sd, ts=r["ts"], win=win, **{k: c[k] for k in c if k not in ("share", "loss", "seed")},
                             share_aaz=share[AAZ], share_apz=share[APZ], loss_aaz=loss[AAZ], loss_apz=loss[APZ],
                             kap_ex_aaz=r["kap_ex"][:, r["fam2idx"][AAZ]], kap_v7tf_aaz=r["kap_v7_tf"][:, r["fam2idx"][AAZ]],
                             kap_ex_apz=r["kap_ex"][:, r["fam2idx"][APZ]], kap_v7tf_apz=r["kap_v7_tf"][:, r["fam2idx"][APZ]]))
            f = lambda x: "  +inf (critical)" if np.isinf(x) else (f"{x:+.3f}" if np.isfinite(x) else "  undef")
            say(f"  {regime:7s} seed {sd}: gate0 {r['gate0']:.1e} | D_share Aaz {share[AAZ]:.3f} (rank {c['rank_aaz']}) "
                f"a+z {share[APZ]:.3f} (rank {c['rank_apz']}) other {other:.1e} | loss Aaz {f(loss[AAZ])} "
                f"a+z {f(loss[APZ])} | err plain {r['err_plain']:.3f} V7 {r['err_v7']:.4f} | V7 stable {stab} "
                f"physical {phys} | E_share(Aaz) {c['E_share_aaz']:.2f} | V7 mag Aaz {mag_aaz:.2f} a+z {mag_apz:.2f}")
    say("\nPER-REGIME DECISION QUANTITIES (pre-registered)")
    verdict_by = {}
    for regime in REGIMES:
        d = decide(cells[regime])
        verdict_by[regime] = d
        say(f"  {regime}: Aaz D_share median {d['med_sh_aaz']:.3f} IQR {d['iqr_sh_aaz'][0]:.3f}-{d['iqr_sh_aaz'][1]:.3f}, "
            f"Aaz > a+z in {d['n_aaz_dom']}/6 -> C1 {d['C1']}")
        say(f"           a+z D_share median {d['med_sh_apz']:.3f} IQR {d['iqr_sh_apz'][0]:.3f}-{d['iqr_sh_apz'][1]:.3f}, "
            f">= 0.15 in {d['n_apz_015']}/6 -> C2 {d['C2']}")
        say(f"           Aaz ablation loss median {d['med_l_aaz']:+.3f} IQR {d['iqr_l_aaz'][0]:+.3f}-{d['iqr_l_aaz'][1]:+.3f}, "
            f"positive in {d['n_l_aaz_pos']}/6 -> C3 {d['C3']}")
        say(f"           a+z ablation loss median {d['med_l_apz']:+.3f} IQR {d['iqr_l_apz'][0]:+.3f}-{d['iqr_l_apz'][1]:+.3f}, "
            f"positive in {d['n_l_apz_pos']}/6 -> C4 {d['C4']}")
    both = lambda k: all(verdict_by[r][k] for r in REGIMES)
    if both("C1") and both("C2") and both("C3") and both("C4"):
        verdict = "CONFIRMED"
    elif both("C1") and both("C3"):
        verdict = "PARTIALLY CONFIRMED"
    else:
        verdict = "NOT CONFIRMED"
    pooled = [c for r in REGIMES for c in cells[r]]
    say(f"\nPOOLED (secondary, 12 cells): Aaz D_share median {np.nanmedian([c['share'][AAZ] for c in pooled]):.3f}; "
        f"a+z {np.nanmedian([c['share'][APZ] for c in pooled]):.3f}; Aaz loss median "
        f"{np.nanmedian([c['loss'][AAZ] for c in pooled]):+.3f}; a+z loss median {np.nanmedian([c['loss'][APZ] for c in pooled]):+.3f}")
    say(f"STRUCTURAL CHECK: max share outside {{Aaz, a+z}} over all cells: "
        f"{max(abs(c.get('other', 0.0)) for c in pooled if not c.get('failed')):.1e} (pre-registered: structurally ~0)")
    say(f"technical failures: {failures or 'none'}")
    say(f"\nPRE-REGISTERED VERDICT: {verdict}")
    say(f"({time.time() - t0:.0f}s)")
    np.savez_compressed("results/mechanism_m5.npz", cells=np.array(slim, dtype=object),
                        decision=np.array(verdict_by, dtype=object), verdict=np.array(verdict),
                        thresholds=np.array(TH, dtype=object))
    with open("results/mechanism_m5.log", "w") as fh:
        fh.write("\n".join(L) + "\n")
    figure(cells)


def figure(cells):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, regime in enumerate(REGIMES):
        cs = [c for c in cells[regime] if not c.get("failed")]
        x = np.arange(len(cs))
        ax[0].scatter(x + 0.1 * i, [c["share"][AAZ] for c in cs], marker="o", label=f"{regime} Aaz")
        ax[0].scatter(x + 0.1 * i, [c["share"][APZ] for c in cs], marker="s", label=f"{regime} a+z")
        la = [min(c["loss"][AAZ], 3.0) for c in cs]
        lp = [min(c["loss"][APZ], 3.0) for c in cs]
        ax[1].scatter(x + 0.1 * i, la, marker="o", label=f"{regime} Aaz (inf shown at 3)")
        ax[1].scatter(x + 0.1 * i, lp, marker="s", label=f"{regime} a+z")
    ax[0].axhline(0.5, ls="--", lw=0.8); ax[0].axhline(0.15, ls=":", lw=0.8)
    ax[0].set_xlabel("draw index (seeds 500-505)"); ax[0].set_ylabel("D_f share"); ax[0].legend(fontsize=7)
    ax[1].axhline(0.5, ls="--", lw=0.8); ax[1].axhline(0.15, ls=":", lw=0.8)
    ax[1].set_xlabel("draw index"); ax[1].set_ylabel("fraction of V7 advantage lost on ablation"); ax[1].legend(fontsize=7)
    fig.suptitle("M = 5 mechanism confirmation (diagnostic plot, not the publication figure)")
    fig.tight_layout()
    fig.savefig("results/mechanism_m5_fig.png", dpi=130)


if __name__ == "__main__":
    if "--regression" in sys.argv:
        regression()
    elif "--campaign" in sys.argv:
        campaign()
    else:
        print(__doc__)
