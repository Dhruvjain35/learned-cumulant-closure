"""
Export the FROZEN V7 closure to a portable, non-executable format (NPZ + JSON) in portable_v7/.

Internal tool: it is the only place that reads the sklearn pickles. The portable package it writes needs neither
sklearn nor pickle. Nothing is refitted or re-serialized through sklearn: every array is copied out of the frozen
estimators exactly (float64, C-contiguous).

Writes:
  portable_v7/models/{v4,v6}/networks.npz   per canonical type: W0,b0,W1,b1,W2,b2 (MLP), mean,scale (scaler),
                                            lo,hi (output clip bounds)
  portable_v7/models/{v4,v6}/metadata.json  architecture, activation, type list, source-pickle sha256
  portable_v7/feature_spec.json             the 21 / 27 inputs in order, ratio transform, canonicalization rules
  portable_v7/router.json                   frozen router constants and inequality direction
  portable_v7/pv7/engine/*.py               the moment engine, vendored from src/engines with ONLY the flat
                                            imports turned into package-relative imports
  portable_v7/metadata.json                 provenance: source commit, hashes, versions

NPZ files are written as stored (uncompressed) zips with a fixed timestamp, so repeated exports are
byte-identical. Run from the repo root:  python run.py src/tools/export_portable_v7.py
"""
import io
import os
import sys
import json
import hashlib
import pickle
import zipfile
import platform
import subprocess
import numpy as np

OUT = "portable_v7"
MODELS = {"v4": "results/lc_v3dens.pkl", "v6": "results/lc_v6.pkl"}
EXPECTED_SHA = {"results/lc_v3dens.pkl": "be2dde48d47be006d97b7aa09c89f7a13e91dc2800fb6dc5cd6ea151fe798f81",
                "results/lc_v6.pkl": "0599643ef05f355f7549e38325261246f0f848fdc71d906f9f0f82b88b800b86"}
DEFINING_SOURCES = ["src/closures/closure_v3.py", "src/closures/closure_v6.py", "src/closures/v7.py",
                    "src/closures/local_closure.py", "src/engines/exact_tc.py", "src/engines/cumulant.py",
                    "src/engines/scalable_engine.py", "src/evaluation/pilot_eval.py", "src/evaluation/local_run.py",
                    "src/evaluation/burst_pred.py"]
VENDORED = {"src/engines/exact_tc.py": [],
            "src/engines/cumulant.py": [("from exact_tc import", "from .exact_tc import")],
            "src/engines/scalable_engine.py": [("from cumulant import", "from .cumulant import")]}
LABEL = {"ad": "ad", "a": "a", "sp": "sp", "sm": "sm", "sz": "sz"}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def type_key(t):
    return "-".join(t)


def write_npz(path, arrays):
    """Deterministic .npz: stored zip, sorted members, fixed timestamp, no pickled objects."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        for k in sorted(arrays):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.ascontiguousarray(arrays[k], dtype=np.float64), allow_pickle=False)
            zi = zipfile.ZipInfo(k + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.external_attr = 0o644 << 16
            z.writestr(zi, buf.getvalue())


def resolve(p):
    if os.path.exists(p):
        return p
    alt = p.replace("src/evaluation/local_run.py", "src/evaluation/local_run.py")
    if os.path.exists(alt):
        return alt
    for d in ("src/closures", "src/engines", "src/evaluation", "src/analysis"):
        q = os.path.join(d, os.path.basename(p))
        if os.path.exists(q):
            return q
    raise FileNotFoundError(p)


FEATURES_21 = [
    dict(col=0, name="single0_re", definition="Re <f0> for canonical factor 0 of the monomial", ratio=False),
    dict(col=1, name="single0_im", definition="Im <f0>", ratio=False),
    dict(col=2, name="single1_re", definition="Re <f1>", ratio=False),
    dict(col=3, name="single1_im", definition="Im <f1>", ratio=False),
    dict(col=4, name="single2_re", definition="Re <f2>", ratio=False),
    dict(col=5, name="single2_im", definition="Im <f2>", ratio=False),
    dict(col=6, name="pair01_re", definition="Re <f0 f1> (normal-ordered pair monomial)", ratio=False),
    dict(col=7, name="pair01_im", definition="Im <f0 f1>", ratio=False),
    dict(col=8, name="pair02_re", definition="Re <f0 f2>", ratio=False),
    dict(col=9, name="pair02_im", definition="Im <f0 f2>", ratio=False),
    dict(col=10, name="pair12_re", definition="Re <f1 f2>", ratio=False),
    dict(col=11, name="pair12_im", definition="Im <f1 f2>", ratio=False),
    dict(col=12, name="det0_over_kappa", definition="(w_site(f0) - omega_c) / kappa; 0 if f0 is a cavity factor", ratio=True),
    dict(col=13, name="det1_over_kappa", definition="(w_site(f1) - omega_c) / kappa; 0 if f1 is a cavity factor", ratio=True),
    dict(col=14, name="det2_over_kappa", definition="(w_site(f2) - omega_c) / kappa; 0 if f2 is a cavity factor", ratio=True),
    dict(col=15, name="nc_per_M", definition="Re <a'a> / M", ratio=False),
    dict(col=16, name="mean_z", definition="mean over i of Re <sz_i>", ratio=False),
    dict(col=17, name="mean_abs_coh", definition="mean over ordered pairs i != j of |<s+_i s-_j>| (0 if M = 1)", ratio=False),
    dict(col=18, name="gc_over_kappa", definition="g_c / kappa (g_c = collective coupling; g_i = g_c/sqrt(M))", ratio=True),
    dict(col=19, name="kappa", definition="kappa (raw, NOT divided)", ratio=False),
    dict(col=20, name="gphi_over_kappa", definition="gamma_phi / kappa", ratio=True),
]
FEATURES_V6_EXTRA = [
    dict(col=21, name="h_nc", definition="memory trace of nc/M (col 15)", ratio=False),
    dict(col=22, name="nc_minus_h_nc", definition="col15 - h_nc", ratio=False),
    dict(col=23, name="h_z", definition="memory trace of mean_z (col 16)", ratio=False),
    dict(col=24, name="z_minus_h_z", definition="col16 - h_z", ratio=False),
    dict(col=25, name="h_coh", definition="memory trace of mean_abs_coh (col 17)", ratio=False),
    dict(col=26, name="coh_minus_h_coh", definition="col17 - h_coh", ratio=False),
]


def feature_spec():
    return dict(
        note=("Machine-readable statement of the features the frozen closures consume, in column order. The ratio "
              "transform divides columns 12, 13, 14, 18 and 20 by column 19 (kappa) AFTER assembly, at train and "
              "apply time identically. Verified against the original code by tests/test_portable_v7.py."),
        v4_width=21, v6_width=27,
        ratio_transform=dict(divide_columns=[12, 13, 14, 18, 20], by_column=19, applied_after_assembly=True),
        columns_v4=FEATURES_21, columns_v6=FEATURES_21 + FEATURES_V6_EXTRA,
        monomial_key=("(p, q, es): normal-ordered a'^p a^q times one emitter operator per site, es[i] in "
                      "{0: identity, 1: s+, 2: s-, 3: sz}"),
        factor_labels=dict(ad="a dagger", a="a", sp="sigma+", sm="sigma-", sz="sigma z (= |e><e| - |g><g|)"),
        canonical_factor_order=("Factors of an order-3 monomial are sorted by (type order ad<a<sp<sm<sz, then site "
                                "index; cavity factors have site -1). Singles are the three factors in that order; "
                                "pairs are (0,1), (0,2), (1,2); detunings use the factors' sites in that order."),
        type_signature="tuple of the three sorted factor labels, e.g. (ad, a, sz)",
        conjugation=dict(
            conj_key="(p, q, es) -> (q, p, es with 1<->2 swapped)",
            routing=("canonical_sig(k): s = sig(k), sc = sig(conj k). If s < sc: predict with k's own features. If "
                     "s > sc: use type sc with conj(k)'s features and conjugate the output. If s == sc: use k's own "
                     "features iff k <= conj(k) (tuple order), else conj(k)'s features and conjugate."),
            self_conjugate="if k == conj(k): imaginary part of the output is set to exactly 0"),
        output=dict(
            components="network output column 0 = Re kappa3, column 1 = Im kappa3 (of the canonical key)",
            clip="each component clipped to [lo, hi] = the per-type training-target min/max, BEFORE conjugation",
            assembly="kappa3(k) = Vc[0] + 1j * csign * Vc[1]; csign = +1 direct, -1 conjugated, 0 self-conjugate",
            missing_type="a monomial whose canonical type has no network gets kappa3 = 0"),
        context_definition="cols 15-17 are computed from the CURRENT tracked moments at every right-hand-side call",
        memory=dict(
            signals="x = (nc/M, mean_z, mean_abs_coh) = cols 15, 16, 17",
            dynamics="dh/dt = lambda * (x - h), integrated as extra ODE state alongside the moments",
            rates="lambda = (kappa, gamma_down, gamma_down/2 + gamma_phi) with gamma_down = 0.02",
            initial="h(0) = x(0) (the context of the initial moments)",
            ratio_transform="memory columns 21-26 are NOT ratio-transformed",
            ode_state="y = [Re m (n_tracked), Im m (n_tracked), h (3)]; solve_ivp RK45, rtol 1e-4, atol 1e-6"),
    )


def main():
    for p, h in EXPECTED_SHA.items():
        got = sha256_file(p)
        assert got == h, f"{p}: sha256 {got} != frozen {h}"
    os.makedirs(OUT, exist_ok=True)
    meta_models = {}
    for branch, pkl in MODELS.items():
        with open(pkl, "rb") as fh:
            lc = pickle.load(fh)
        types = sorted(lc.nets)
        arrays, tinfo = {}, []
        for t in types:
            net, sc = lc.nets[t], lc.scalers[t]
            assert net.activation == "relu" and net.out_activation_ == "identity", (t, net.activation)
            assert sc.with_mean and sc.with_std
            tk = type_key(t)
            for i, (W, b) in enumerate(zip(net.coefs_, net.intercepts_)):
                arrays[f"{tk}__W{i}"] = W
                arrays[f"{tk}__b{i}"] = b
            arrays[f"{tk}__mean"] = sc.mean_
            arrays[f"{tk}__scale"] = sc.scale_
            lo, hi = lc.range[t]
            arrays[f"{tk}__lo"] = lo
            arrays[f"{tk}__hi"] = hi
            tinfo.append(dict(type=list(t), key=tk, layers=[list(W.shape) for W in net.coefs_],
                              n_features_in=int(net.n_features_in_), n_outputs=int(net.n_outputs_)))
        d = os.path.join(OUT, "models", branch)
        os.makedirs(d, exist_ok=True)
        write_npz(os.path.join(d, "networks.npz"), arrays)
        bm = dict(branch=branch, source_pickle=pkl, source_pickle_sha256=EXPECTED_SHA[pkl],
                  estimator="sklearn MLPRegressor + StandardScaler (exported arrays only; no sklearn needed)",
                  hidden_activation="relu", output_activation="identity",
                  forward=("z = (x - mean) / scale; h1 = relu(z @ W0 + b0); h2 = relu(h1 @ W1 + b1); "
                           "y = h2 @ W2 + b2; y_clipped = clip(y, lo, hi) componentwise"),
                  input_width=tinfo[0]["n_features_in"], n_types=len(types), dtype="float64", types=tinfo)
        with open(os.path.join(d, "metadata.json"), "w") as fh:
            json.dump(bm, fh, indent=1)
        meta_models[branch] = dict(npz_sha256=sha256_file(os.path.join(d, "networks.npz")), n_types=len(types),
                                   input_width=bm["input_width"])
    with open(os.path.join(OUT, "feature_spec.json"), "w") as fh:
        json.dump(feature_spec(), fh, indent=1)
    from v7 import BOX_KA, BOX_GP
    router = dict(rule="branch = 'v6' if (kappa <= kappa_max and gamma_phi <= gamma_phi_max) else 'v4'",
                  kappa_max=BOX_KA, gamma_phi_max=BOX_GP, inequality="<= (inclusive) on both",
                  decided="once at t = 0 from (kappa, gamma_phi); no blending, no switching",
                  v4_model="models/v4 (frozen lc_v3dens.pkl)", v6_model="models/v6 (frozen lc_v6.pkl)")
    with open(os.path.join(OUT, "router.json"), "w") as fh:
        json.dump(router, fh, indent=1)
    eng_dir = os.path.join(OUT, "pv7", "engine")
    os.makedirs(eng_dir, exist_ok=True)
    with open(os.path.join(eng_dir, "__init__.py"), "w") as fh:
        fh.write('"""Moment engine vendored verbatim from src/engines/ (only flat imports made relative)."""\n')
    vend = {}
    for src, subs in VENDORED.items():
        txt = open(src).read()
        for a, b in subs:
            assert txt.count(a) >= 1, (src, a)
            txt = txt.replace(a, b)
        dst = os.path.join(eng_dir, os.path.basename(src))
        with open(dst, "w") as fh:
            fh.write(txt)
        vend[src] = dict(source_sha256=sha256_file(src), vendored_sha256=sha256_file(dst), substitutions=subs)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--"] + [resolve(p) for p in DEFINING_SOURCES] + list(EXPECTED_SHA)).decode().strip())
    except Exception:
        commit, dirty = "unknown", None
    import sklearn
    meta = dict(
        package="portable frozen V7 (learned third-cumulant closure); export only, no retraining",
        source_repo_commit=commit,
        source_tree_dirty_at_export=dirty,
        source_pickles={p: h for p, h in EXPECTED_SHA.items()},
        models=meta_models,
        defining_sources={p: sha256_file(resolve(p)) for p in DEFINING_SOURCES},
        vendored_engine=vend,
        feature_spec_sha256=sha256_file(os.path.join(OUT, "feature_spec.json")),
        router_sha256=sha256_file(os.path.join(OUT, "router.json")),
        export_script_sha256=sha256_file(__file__),
        export_environment=dict(python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__),
    )
    with open(os.path.join(OUT, "metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(json.dumps(meta_models, indent=1))
    print("export complete")


def write_sums():
    """SHA256SUMS over every package file (code, models, specs, fixture), excluding run outputs and caches."""
    rows = []
    for d, dirs, files in os.walk(OUT):
        dirs[:] = sorted(x for x in dirs if x not in ("__pycache__", "external_output", "selfcheck_output"))
        for f in sorted(files):
            if f == "SHA256SUMS" or f.endswith(".pyc"):
                continue
            p = os.path.join(d, f)
            rows.append(f"{sha256_file(p)}  {os.path.relpath(p, OUT)}")
    with open(os.path.join(OUT, "SHA256SUMS"), "w") as fh:
        fh.write("\n".join(sorted(rows, key=lambda r: r.split("  ")[1])) + "\n")
    print(f"SHA256SUMS: {len(rows)} files")


if __name__ == "__main__":
    if "--sums-only" not in sys.argv:
        main()
    write_sums()
