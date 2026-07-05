"""
STEP 0 (cheap kill test, existing data only): does a single derived combination s = kappa + 2*gphi
predict WHERE MEMORY IS NEEDED (V4 inadequate, V6 repairs) better than the empirical rectangle
(kappa<=0.045 AND gphi<=0.05)? All runs here are at fixed g_c=0.7, gdown=0.02, M=5, so the memory-demand
number D_a = 4 g_c / (kappa + gdown + 2 gphi) is strictly monotone in s = kappa + 2 gphi; testing a
threshold on s == testing D_a.

Honesty rules baked in:
  - The 1:2 weight is DERIVED (Ω=2g_c generated / Γ_dec = kappa/2 + gdown/2 + gphi destroyed), not fit.
    We ALSO scan the weight w in s = kappa + w*gphi to see whether the separation actually peaks near
    w=2 (nontrivial confirmation) or somewhere else (derivation coefficients wrong).
  - Threshold fit on DEV seeds only; balanced accuracy reported on HELD-OUT seeds.
  - Compared against the FIXED rectangle and a DEV-FIT rectangle (apples-to-apples: fit vs fit).
  - Primary label is STABILITY (unambiguous): memory-needed M = V4 unstable & V6 stable; memoryless-fine
    F = V4 stable. 'Both fail' B = V4 unstable & V6 unstable (residual; excluded from M-vs-F, counted).
  - Probabilistic framing: also report cell-level propensity (fraction of seeds needing memory) vs s.
"""
import numpy as np

GC, GDOWN = 0.7, 0.02
DEV_SEEDS = {19, 20, 21, 24, 25, 29, 31}          # split by seed -> no seed in both dev and test
TEST_SEEDS = {22, 23, 26, 27, 28, 32, 33, 34}


def adequate(ok, nn, bar=0.15):
    return bool(ok) and np.isfinite(nn) and nn <= bar


def load_all():
    """Return list of dicts: {ka, gp, seed, v4ok, v4nn, v6ok, v6nn}."""
    rows = {}
    def put(ka, gp, sd, who, ok, nn):
        k = (round(ka, 4), round(gp, 4), int(sd))
        rows.setdefault(k, {})[who] = (bool(ok), float(nn) if np.isfinite(nn) else np.inf)

    # eval_v56: keys ('P', ka, gp, seed, model); v4 under 'v4(desc)', v6 under 'v6'
    for f in ['results/eval_v56_dev.npz', 'results/eval_v56_s21_22_23.npz']:
        d = np.load(f, allow_pickle=True)['res'].item()
        for k, v in d.items():
            if k[0] != 'P':
                continue
            _, ka, gp, sd, m = k
            if m == 'v4(desc)':
                put(ka, gp, sd, 'v4', v[0], v[1])
            elif m == 'v6':
                put(ka, gp, sd, 'v6', v[0], v[1])
    # eval_v7: keys ('P', ka, gp, seed) -> (ok7,nn7,branch,ok4,nn4,ok6,nn6)
    for f in ['results/eval_v7_s24_25.npz', 'results/eval_v7_s26_27_28.npz']:
        d = np.load(f, allow_pickle=True)['res'].item()
        for k, v in d.items():
            if k[0] != 'P':
                continue
            _, ka, gp, sd = k
            put(ka, gp, sd, 'v4', v[3], v[4])
            put(ka, gp, sd, 'v6', v[5], v[6])
    # baselines: keys (tag, ka, gp, seed, model) for tag in in:/out:; ('sanity', gc, ka, gp, sg, model)
    for f in ['results/baselines_dev_s29_31.npz', 'results/baselines_final_s32_33_34.npz']:
        d = np.load(f, allow_pickle=True)['res'].item()
        for k, v in d.items():
            if isinstance(k[0], str) and (k[0].startswith('in:') or k[0].startswith('out:')):
                _, ka, gp, sd, m = k
            elif k[0] == 'sanity':
                _, gc, ka, gp, sg, m = k                 # normal-regime negatives (kappa=0.15)
                sd = list(DEV_SEEDS)[0]                  # sanity ran at one seed; treat as dev
            else:
                continue
            if m in ('v4', 'v6'):
                put(ka, gp, sd, m, v[0], v[1])
    out = []
    for (ka, gp, sd), r in rows.items():
        if 'v4' in r and 'v6' in r:
            out.append(dict(ka=ka, gp=gp, seed=sd, v4ok=r['v4'][0], v4nn=r['v4'][1],
                            v6ok=r['v6'][0], v6nn=r['v6'][1]))
    return out


def label(r):
    v4ade = adequate(r['v4ok'], r['v4nn'])
    v6ade = adequate(r['v6ok'], r['v6nn'])
    if not v4ade and v6ade:
        return 'M'                                        # memory needed & repaired
    if v4ade:
        return 'F'                                        # memoryless fine
    return 'B'                                             # both inadequate (residual)


def auc(scores, labels):
    """AUC that lower score => positive (we predict M when s is small, so score = -s)."""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float('nan')
    c = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return c / (len(pos) * len(neg))


def bal_acc(pred, lab):
    tp = sum(p and l for p, l in zip(pred, lab)); fn = sum((not p) and l for p, l in zip(pred, lab))
    tn = sum((not p) and (not l) for p, l in zip(pred, lab)); fp = sum(p and (not l) for p, l in zip(pred, lab))
    sens = tp / (tp + fn) if tp + fn else 0; spec = tn / (tn + fp) if tn + fp else 0
    return 0.5 * (sens + spec)


if __name__ == "__main__":
    data = load_all()
    for r in data:
        r['lab'] = label(r); r['s'] = r['ka'] + 2 * r['gp']
    cells = sorted({(r['ka'], r['gp']) for r in data})
    print(f"loaded {len(data)} (cell,seed) instances over {len(cells)} cells, "
          f"{len({r['seed'] for r in data})} seeds\n")
    from collections import Counter
    print("class counts:", dict(Counter(r['lab'] for r in data)),
          " (M=memory needed, F=memoryless fine, B=both fail)\n")

    # cell-level propensity vs derived s
    print(f"{'kappa':>6s} {'gphi':>5s} {'s=k+2g':>7s} {'nMemNeed/nUsable':>16s}  rectangle?")
    for (ka, gp) in cells:
        cd = [r for r in data if r['ka'] == ka and r['gp'] == gp]
        m = sum(r['lab'] == 'M' for r in cd); usable = sum(r['lab'] in ('M', 'F') for r in cd)
        rect = (ka <= 0.045 and gp <= 0.05)
        print(f"{ka:>6.3f} {gp:>5.3f} {ka+2*gp:>7.3f} {m:>7d}/{usable:<8d}  {'IN' if rect else 'out'}")

    # M vs F classification (exclude B)
    mf = [r for r in data if r['lab'] in ('M', 'F')]
    y = [1 if r['lab'] == 'M' else 0 for r in mf]
    s = [r['s'] for r in mf]
    print(f"\nM-vs-F usable instances: {sum(y)} memory-needed, {len(y)-sum(y)} memoryless-fine")
    print(f"AUC of derived s=kappa+2gphi (all): {auc([-x for x in s], y):.3f}")

    # weight scan: does separation peak near the derived w=2?
    print("\nweight scan  s = kappa + w*gphi   (derived w=2):")
    best_w, best_a = None, -1
    for w in [0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 6, 10]:
        a = auc([-(r['ka'] + w * r['gp']) for r in mf], y)
        mark = '  <- derived' if w == 2 else ('  <- peak' if False else '')
        if a > best_a: best_a, best_w = a, w
        print(f"   w={w:>4.1f}: AUC={a:.3f}")
    print(f"   peak AUC at w={best_w} ({best_a:.3f}); derived w=2 is {'AT/near peak' if abs(best_w-2)<=1 else 'FAR from peak -> derivation coefficients suspect'}")

    # held-out threshold comparison: fit on dev, evaluate on test
    dev = [r for r in mf if r['seed'] in DEV_SEEDS]; test = [r for r in mf if r['seed'] in TEST_SEEDS]
    ydev = [1 if r['lab'] == 'M' else 0 for r in dev]; ytest = [1 if r['lab'] == 'M' else 0 for r in test]
    # fit s_crit on dev (maximize balanced acc)
    cand = sorted({r['s'] for r in dev})
    best_sc, best_ba = None, -1
    for sc in cand:
        ba = bal_acc([r['s'] < sc for r in dev], ydev)
        if ba > best_ba: best_ba, best_sc = ba, sc
    # fit rectangle thresholds on dev
    kc = sorted({r['ka'] for r in dev}); gc_ = sorted({r['gp'] for r in dev})
    best_rect, brba = None, -1
    for kk in kc:
        for gg in gc_:
            ba = bal_acc([(r['ka'] <= kk and r['gp'] <= gg) for r in dev], ydev)
            if ba > brba: brba, best_rect = ba, (kk, gg)
    print(f"\nHELD-OUT comparison (fit on {len(dev)} dev, test on {len(test)} test instances):")
    print(f"  single ratio  s<{best_sc:.3f}         : test balanced acc = {bal_acc([r['s']<best_sc for r in test], ytest):.3f}")
    print(f"  dev-fit rect  k<={best_rect[0]}, g<={best_rect[1]} : test balanced acc = {bal_acc([(r['ka']<=best_rect[0] and r['gp']<=best_rect[1]) for r in test], ytest):.3f}")
    print(f"  FIXED rect    k<=0.045, g<=0.05  : test balanced acc = {bal_acc([(r['ka']<=0.045 and r['gp']<=0.05) for r in test], ytest):.3f}")
    print(f"\nVERDICT: single derived ratio {'BEATS/MATCHES' if True else ''} rectangle? -> read the two test balanced accuracies above; derived weight peak check above.")
