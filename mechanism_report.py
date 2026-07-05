"""
Decision + table + figure for the frozen mechanism analysis (reads results/mechanism.npz). Applies the
PRE-REGISTERED decision bar (PROTOCOL "MECHANISM ANALYSIS PRE-REGISTRATION"): a family is a mechanistic driver
iff it (1) dominates the non-circular D_f share (>=0.5 AND >2x the second), (2) V7 reproduces sign+magnitude,
(3) its ablation removes a substantial share of V7's burst-peak advantage (frac_lost>=0.5 or destabilizes V7),
consistently across seeds 37/38/39. Conclusion 1 (single) / 2 (small-family distributed) / 3 (broad).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PHYS = {"Aaz": "photon-number-inversion  <a†a σz>",
        "a+z": "photon-polarization-inversion  <a σ⁺ σz>",
        "Aa-": "photon-number-coherence  <a†a σ⁻>",
        "+-z": "spin-exchange-inversion  <σ⁺σ⁻ σz>",
        "a+-": "photon-spin-exchange  <a σ⁺σ⁻>"}
SEEDS = [37, 38, 39]


def load():
    return np.load("results/mechanism.npz", allow_pickle=True)["res"].item()


def famlabel(f):
    return "".join({"ad": "A", "a": "a", "sp": "+", "sm": "-", "sz": "z"}[t] for t in f)


def v7_repro(res, regime, f):
    """D-based magnitude ratio and alignment sign of teacher-forced V7 kappa3 vs exact, on family f keys."""
    mags, signs = [], []
    for s in SEEDS:
        r = res[(regime, s)]
        idx = r["fam2idx"][f]; win = r["win"]
        ex = r["kap_ex"][np.ix_(win, idx)]; v7 = r["kap_v7_tf"][np.ix_(win, idx)]
        ne = np.linalg.norm(ex)
        if ne > 1e-10:
            mags.append(np.linalg.norm(v7) / ne)
            signs.append(np.sign(np.real(np.sum(np.conj(ex) * v7))))
    return (float(np.mean(mags)) if mags else np.nan, float(np.mean(signs)) if signs else np.nan)


def frac_lost_agg(res, regime, f):
    vals = [res[(regime, s)]["frac_lost"].get(f, np.nan) for s in SEEDS]
    ncrit = sum(np.isinf(x) for x in vals)
    fin = [x for x in vals if np.isfinite(x)]
    return (float(np.mean(fin)) if fin else np.nan), ncrit


def decide(res, regime):
    fams = res[(regime, SEEDS[0])]["fams"]
    shareD = res[(regime, "shareD")]
    rankedD = res[(regime, "rankedD")]
    rows = []
    for f in rankedD:
        fl, ncrit = frac_lost_agg(res, regime, f)
        mag, sgn = v7_repro(res, regime, f)
        rows.append(dict(f=f, shareD=shareD[f], shareE=res[(regime, "shareE")][f],
                         fl=fl, ncrit=ncrit, mag=mag, sgn=sgn))
    top, second = rows[0], rows[1] if len(rows) > 1 else None
    # pre-registered single-family bar
    dominates = top["shareD"] >= 0.5 and (second is None or top["shareD"] > 2 * second["shareD"])
    v7ok = (0.5 <= top["mag"] <= 2.0) and top["sgn"] > 0
    ablbig = (np.isfinite(top["fl"]) and top["fl"] >= 0.5) or top["ncrit"] >= 2
    single = dominates and v7ok and ablbig
    # small-family: top-2 cumulative shareD >= 0.9 and second is non-trivial (>=0.15 shareD or fl>=0.15)
    cum2 = top["shareD"] + (second["shareD"] if second else 0)
    secreal = second is not None and (second["shareD"] >= 0.15 or (np.isfinite(second["fl"]) and second["fl"] >= 0.15))
    if single:
        concl = 1
    elif cum2 >= 0.9 and secreal:
        concl = 2
    else:
        concl = 3
    return rows, concl, dict(dominates=dominates, v7ok=v7ok, ablbig=ablbig, cum2=cum2)


def rep_key(res, regime, f):
    """representative key index (in eng_o3 order) for family f: the first key of that family."""
    r = res[(regime, SEEDS[0])]
    return r["fam2idx"][f][0]


def fig(res):
    fams_order = res[("POCKET", "rankedD")]
    regs = ["POCKET", "CONTROL"]
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25})
    figu, ax = plt.subplots(2, 3, figsize=(13, 7))
    for ri, regime in enumerate(regs):
        rankedD = res[(regime, "rankedD")]
        shareD = res[(regime, "shareD")]
        # (a) D_f family-share bars (top 5)
        top5 = rankedD[:5]
        ax[ri, 0].bar([famlabel(f) for f in top5], [shareD[f] for f in top5], color="#3b6ea5")
        ax[ri, 0].set_ylabel(f"{regime}\nD_f share"); ax[ri, 0].set_ylim(0, 1)
        ax[ri, 0].set_title("(a) which κ3 families carry the correction" if ri == 0 else "")
        # (b) dominant-family representative-key kappa3(t) in burst window: exact / V7 / T3 / plain
        f0 = rankedD[0]; ki = rep_key(res, regime, f0)
        s = SEEDS[0]; r = res[(regime, s)]; win = r["win"]; ts = r["ts"][win]
        ex = np.real(r["kap_ex"][win, ki]); v7 = np.real(r["kap_v7_cl"][win, ki])
        ax[ri, 1].plot(ts, ex, "k-", lw=2, label="exact κ3")
        ax[ri, 1].plot(ts, v7, "-", color="#2a9d8f", lw=1.8, label="V7")
        if r["kap_t3"] is not None:
            ax[ri, 1].plot(ts, np.real(r["kap_t3"][win, ki]), ":", color="#e76f51", lw=1.8, label="κ4=0 T3")
        else:
            ax[ri, 1].text(0.5, 0.9, "T3 divergent", transform=ax[ri, 1].transAxes, ha="center", color="#e76f51")
        ax[ri, 1].axhline(0, color="#999", lw=1.2, ls="--", label="plain (κ3=0)")
        ax[ri, 1].set_ylabel("Re κ3 (dom. family)")
        ax[ri, 1].set_title(f"(b) {famlabel(f0)} = {PHYS.get(famlabel(f0),'')}" if ri == 0 else f"{famlabel(f0)}")
        if ri == 0:
            ax[ri, 1].legend(fontsize=7, loc="best")
        ax[ri, 1].set_xlabel("t")
        # (c) ablation frac_lost bars (top 5)
        fls = []
        for f in top5:
            fl, ncrit = frac_lost_agg(res, regime, f)
            fls.append(1.2 if ncrit else (fl if np.isfinite(fl) else 0))
        cols = ["#c1121f" if (frac_lost_agg(res, regime, f)[1] > 0) else "#3b6ea5" for f in top5]
        ax[ri, 2].bar([famlabel(f) for f in top5], fls, color=cols)
        ax[ri, 2].axhline(1.0, color="#333", ls="--", lw=1, label="removes all V7 advantage")
        ax[ri, 2].set_ylabel("frac. of V7 advantage lost")
        ax[ri, 2].set_title("(c) closed-loop ablation (red = destabilizes V7)" if ri == 0 else "")
        if ri == 0:
            ax[ri, 2].legend(fontsize=7)
    figu.suptitle("Superradiant burst-peak correction: connected third-cumulant mechanism (frozen V7, exact-referenced)",
                  fontsize=11)
    figu.tight_layout(rect=[0, 0, 1, 0.97])
    figu.savefig("results/mechanism_fig.png", dpi=140)
    print("saved results/mechanism_fig.png")


if __name__ == "__main__":
    res = load()
    concls = {}
    for regime in ["POCKET", "CONTROL"]:
        rows, concl, why = decide(res, regime)
        concls[regime] = concl
        print(f"\n=== {regime} (M{res[(regime, SEEDS[0])]['kap_ex'].shape[0] and ''}) ===")
        print(f"  {'family':7s} {'physical':38s} {'D_sh':>5s} {'E_sh':>5s} {'fracLost':>9s} {'V7mag':>6s} {'V7sgn':>6s}")
        for r in rows[:6]:
            lab = famlabel(r["f"])
            crit = f" CRIT×{r['ncrit']}" if r["ncrit"] else ""
            fl = "unstable" if r["ncrit"] >= 2 else (f"{r['fl']:+.2f}" if np.isfinite(r["fl"]) else "  n/a")
            print(f"  {lab:7s} {PHYS.get(lab,''):38s} {r['shareD']:5.2f} {r['shareE']:5.2f} {fl:>9s} "
                  f"{r['mag']:6.2f} {r['sgn']:+6.1f}{crit}")
        print(f"  -> conclusion {concl}  ({why})")
    overall = 2 if set(concls.values()) == {1, 2} else (concls["POCKET"] if len(set(concls.values())) == 1 else max(concls.values()))
    names = {1: "SINGLE-FAMILY mechanism", 2: "SMALL-FAMILY DISTRIBUTED mechanism",
             3: "BROADLY DISTRIBUTED (no compact attribution)"}
    print(f"\nOVERALL PRE-REGISTERED CONCLUSION: {overall} = {names[overall]}")
    fig(res)
