"""
SELF-CHECK FIXTURE -- NOT AN INDEPENDENT RESULT.

Run this first, from this directory:   python run_selfcheck.py
It checks that this copy of the package and this Python environment reproduce the frozen originals before any
panel cell is run. Expected values (selfcheck_expected.json) were produced by the original sklearn-pickle stack.

Checks and tolerances (fixed in advance; they allow for cross-machine floating-point differences, while on the
authors' machine every difference is exactly 0):
  1. SHA256SUMS: every package file hashes as recorded                        exact
  2. disorder RNG: draws for the published seed 50 equal the recorded ones     exact (bit for bit)
  3. static network outputs (16 inputs per branch, before clipping)            max abs diff <= 1e-9
  4. exact reference burst peak (QuTiP mesolve, M = 2)                         rel diff <= 1e-5
  5. plain second-order burst peak                                             rel diff <= 1e-4
  6. V7 branch, burst peak, and n(t)                                           same branch; peak rel diff <= 1e-4;
                                                                               max |n(t) diff| <= 1e-3
  7. scikit-learn / joblib never imported                                      exact
Exit status 0 = PASS. Writes selfcheck_output/selfcheck.json.
"""
import os
import sys
import json
import hashlib
import platform
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def run_selfcheck(verbose=True):
    import scipy
    import qutip
    import pv7.rollout as pr
    from pv7.engine.exact_tc import draw_disorder
    res, ok_all = [], True

    def rec(name, ok, detail):
        nonlocal ok_all
        ok_all &= bool(ok)
        res.append(dict(check=name, ok=bool(ok), detail=detail))
        if verbose:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    bad = []
    with open(os.path.join(HERE, "SHA256SUMS")) as fh:
        for line in fh:
            h, rel = line.rstrip("\n").split("  ", 1)
            p = os.path.join(HERE, rel)
            if not os.path.exists(p) or sha256(p) != h:
                bad.append(rel)
    rec("SHA256SUMS", not bad, f"{len(bad)} mismatched files {bad[:5]}")

    exp = json.load(open(os.path.join(HERE, "selfcheck_expected.json")))
    for c in exp["cases"]:
        w = draw_disorder(c["M"], 1.0, c["sigma"], c["seed"], "gauss")
        rec(f"{c['id']} disorder draw (seed {c['seed']})", np.array_equal(w, np.array(c["w"])),
            f"max |diff| {float(np.max(np.abs(w - np.array(c['w'])))):.1e}")

    worst = 0.0
    for s in exp["static"]:
        net = pr.branch(s["branch"]).nets[tuple(s["type"])]
        worst = max(worst, float(np.max(np.abs(net.predict_raw(np.array(s["X"])) - np.array(s["y_raw"])))))
    rec("static network outputs", worst <= 1e-9, f"max abs diff {worst:.1e}")

    for c in exp["cases"]:
        M, gc, ka, gp, sg, seed = c["M"], c["g_c"], c["kappa"], c["gamma_phi"], c["sigma"], c["seed"]
        eng, w, g = pr.build_cell(M, gc, ka, gp, sg, seed)
        ts, ne, _, _ = pr.exact_ref(M, w, g, ka, gp)
        ref = pr.burst_obs(ts, ne, ka)
        rec(f"{c['id']} exact burst peak", abs(ref["npeak"] - c["exact_npeak"]) <= 1e-5 * c["exact_npeak"],
            f"{ref['npeak']:.8f} vs {c['exact_npeak']:.8f}")
        m0 = pr.m0_product(eng, 1.0)
        _, tp = pr.run_from(eng, m0, None)
        npl, _ = pr.observables(eng, tp)
        rec(f"{c['id']} plain burst peak", abs(np.max(npl) - c["plain_npeak"]) <= 1e-4 * c["plain_npeak"],
            f"{np.max(npl):.8f} vs {c['plain_npeak']:.8f}")
        br, t7 = pr.run_v7(eng, m0, w - pr.WC, (gc, ka, gp))
        n7, _ = pr.observables(eng, t7)
        dn = float(np.max(np.abs(n7 - np.array(c["v7_n_t"]))))
        rec(f"{c['id']} V7 ({br})", br == c["branch"] and abs(np.max(n7) - c["v7_npeak"]) <= 1e-4 * c["v7_npeak"]
            and dn <= 1e-3, f"branch {br}, peak {np.max(n7):.8f} vs {c['v7_npeak']:.8f}, max |n(t) diff| {dn:.1e}")

    leaked = sorted(m for m in sys.modules if m.split(".")[0] in ("sklearn", "joblib"))
    rec("no scikit-learn / joblib imported", not leaked, str(leaked))
    env = dict(python=platform.python_version(), platform=platform.platform(), numpy=np.__version__,
               scipy=scipy.__version__, qutip=qutip.__version__)
    os.makedirs(os.path.join(HERE, "selfcheck_output"), exist_ok=True)
    with open(os.path.join(HERE, "selfcheck_output", "selfcheck.json"), "w") as fh:
        json.dump(dict(label=exp["label"], passed=ok_all, environment=env, checks=res), fh, indent=1)
    if verbose:
        print(f"\nSELF-CHECK {'PASS' if ok_all else 'FAIL'}  ({env})")
    return ok_all, env


if __name__ == "__main__":
    print("SELF-CHECK FIXTURE -- NOT AN INDEPENDENT RESULT")
    ok, _ = run_selfcheck()
    sys.exit(0 if ok else 1)
