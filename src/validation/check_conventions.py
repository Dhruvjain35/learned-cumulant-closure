"""
Convention gates: turn the model's stated conventions into TESTED claims.

`validate_model.py` already gates the coherent part of H against analytic limits (vacuum Rabi
wc +/- g at M=1; collective splitting 2 g sqrt(M)). What it does not touch is the DISSIPATIVE
side, and that is where the project's one unchecked assumption lived: the pure-dephasing form
(gphi/2) D[sz], the claim that it produces transverse decay gamma_perp = gdown/2 + gphi, and the
excited-state sign convention that H and the fully-inverted initial state both depend on.

Every gate below compares against a CLOSED-FORM result, not against another part of this codebase.

  C1  operators rebuilt from scratch (raw numpy, from the model equation as written in the docs)
      equal the ones exact_tc.py constructs, to machine precision
  C2  excited-state sign conventions: sz = +1 on excited, H's emitter term costs w on excited,
      and the burst initial state is genuinely fully inverted
  C3  cavity loss: <a'a> decays at kappa, <a> at kappa/2
  C4  emitter decay: <sp sm> decays at gdown, |<sm>| at gdown/2
  C5  pure dephasing: |<sm>| decays at gphi, population is untouched
  C6  combined: |<sm>| decays at gdown/2 + gphi. This is gamma_perp, the Result 23 input.
  C7  the same rates hold inside the CumulantEngine, so the convention is not merely correct in
      the exact solver but carried identically into the closure engine

HONEST LIMITATION. C1 rebuilds the operators from the documented model equation, but the same
person wrote both the documentation and the code, so C1 rules out a transcription error, not a
shared conceptual one. C2-C7 are stronger: they check the model against closed-form physics that
was not derived from this codebase. What no self-check can establish is that the documented model
is the one a reader means by "open disordered Tavis-Cummings"; that is a human read, and it is
tracked in docs/EXTERNAL_VALIDATION_STATUS.md.
"""
import numpy as np
import qutip as qt

from exact_tc import build_ops, build_H, collapse_ops
from cumulant import CumulantEngine

TOL = 1e-9


def _scratch_ops(M, Nc):
    """Rebuild the operators from the model equation using raw numpy, independent of exact_tc.

    Convention taken from the documented model: index 0 of each two-level factor is the EXCITED
    state, the tensor order is cavity-first, and sm = |g><e| lowers.
    """
    a1 = np.diag(np.sqrt(np.arange(1, Nc)), 1)          # <n|a|n+1> = sqrt(n+1)
    Ic = np.eye(Nc)
    e = np.array([1.0, 0.0])                             # excited
    g = np.array([0.0, 1.0])                             # ground
    sm1 = np.outer(g, e)                                 # |g><e|
    I2 = np.eye(2)

    def kron_all(mats):
        out = np.array([[1.0]])
        for m in mats:
            out = np.kron(out, m)
        return out

    a = kron_all([a1] + [I2] * M)
    sm, sp, sz, ee = [], [], [], []
    for i in range(M):
        mats = [Ic] + [I2] * M
        mats[i + 1] = sm1
        smi = kron_all(mats)
        sm.append(smi)
        sp.append(smi.conj().T)
        ee.append(sp[-1] @ smi)
        sz.append(sp[-1] @ smi - smi @ sp[-1])
    return {"a": a, "sm": sm, "sp": sp, "sz": sz, "ee": ee}


def _rate_from_closed_form(times, values, rate, label, tol=TOL):
    """values(t) must equal values(0) * exp(-rate * t) for every t."""
    pred = values[0] * np.exp(-rate * np.asarray(times))
    err = float(np.max(np.abs(values - pred)))
    ok = err < tol
    print(f"   {label:<52} max|measured - closed form| = {err:.3e}  {'PASS' if ok else 'FAIL'}")
    return ok


def _decay_run(H, c_ops, psi0, obs, times):
    # tolerances well below the gate threshold: these gates test the model, not the integrator
    res = qt.mesolve(H, psi0, times, c_ops=c_ops, e_ops=[],
                     options={"progress_bar": False, "atol": 1e-13, "rtol": 1e-11})
    return np.array([qt.expect(obs, s) for s in res.states], complex)


def gate1_scratch_operators():
    print("\n[C1] operators rebuilt from the documented equation == exact_tc construction")
    M, Nc = 2, 6
    ops = build_ops(M, Nc)
    ref = _scratch_ops(M, Nc)
    ok = True
    checks = [("a", ops["a"].full(), ref["a"])]
    for name in ("sm", "sp", "sz", "ee"):
        for i in range(M):
            checks.append((f"{name}[{i}]", ops[name][i].full(), ref[name][i]))
    worst = 0.0
    for name, A, B in checks:
        d = float(np.max(np.abs(A - B)))
        worst = max(worst, d)
        ok = ok and d < 1e-14
    print(f"   {len(checks)} operators, max elementwise difference = {worst:.3e}  {'PASS' if ok else 'FAIL'}")

    wc, w, g = 1.0, np.array([1.0, 1.05]), np.array([0.2, 0.3])
    H = build_H(ops, wc, w, g).full()
    Hs = wc * (ref["a"].conj().T @ ref["a"])
    for i in range(M):
        Hs = Hs + w[i] * ref["ee"][i] + g[i] * (ref["a"] @ ref["sp"][i] + ref["a"].conj().T @ ref["sm"][i])
    dH = float(np.max(np.abs(H - Hs)))
    okH = dH < 1e-14
    print(f"   {'H rebuilt from the model equation':<52} max difference = {dH:.3e}  {'PASS' if okH else 'FAIL'}")
    return ok and okH


def gate2_sign_conventions():
    print("\n[C2] excited-state sign conventions (H, sz, and the burst initial state)")
    M, Nc = 2, 6
    ops = build_ops(M, Nc)
    exc = qt.tensor(qt.basis(Nc, 0), qt.basis(2, 0), qt.basis(2, 0))   # the burst IC: vacuum, all excited
    gnd = qt.tensor(qt.basis(Nc, 0), qt.basis(2, 1), qt.basis(2, 1))
    ok = True
    for i in range(M):
        v_e = qt.expect(ops["sz"][i], exc).real
        v_g = qt.expect(ops["sz"][i], gnd).real
        p = abs(v_e - 1.0) < 1e-12 and abs(v_g + 1.0) < 1e-12
        ok = ok and p
        print(f"   emitter {i}: <sz> = {v_e:+.1f} excited, {v_g:+.1f} ground             "
              f"{'PASS' if p else 'FAIL'}")
    wc, w, g = 1.0, np.array([1.0, 1.05]), np.array([0.0, 0.0])
    H = build_H(ops, wc, w, g)
    E_e = qt.expect(H, exc).real
    E_g = qt.expect(H, gnd).real
    p = abs(E_e - w.sum()) < 1e-12 and abs(E_g) < 1e-12
    ok = ok and p
    print(f"   H(all excited) = {E_e:.4f} (expected sum w = {w.sum():.4f}); H(all ground) = {E_g:.4f}"
          f"   {'PASS' if p else 'FAIL'}")
    n0 = qt.expect(ops["a"].dag() * ops["a"], exc).real
    tot = sum(qt.expect(ops["ee"][i], exc).real for i in range(M))
    p = abs(n0) < 1e-12 and abs(tot - M) < 1e-12
    ok = ok and p
    print(f"   burst IC: <a'a> = {n0:.1e}, total excited population = {tot:.4f} (expected M = {M})"
          f"        {'PASS' if p else 'FAIL'}")
    return ok


def gate3_cavity_loss():
    print("\n[C3] cavity loss: <a'a> decays at kappa, <a> at kappa/2")
    Nc, kappa = 12, 0.17
    ops = build_ops(0, Nc)
    a = ops["a"]
    H = 0.0 * a.dag() * a                                  # frame with wc = 0 so <a> is real-decaying
    c = collapse_ops(ops, kappa=kappa)
    alpha = 0.8
    psi0 = qt.coherent(Nc, alpha)
    t = np.linspace(0, 6, 13)
    n = _decay_run(H, c, psi0, a.dag() * a, t).real
    amp = _decay_run(H, c, psi0, a, t)
    ok = _rate_from_closed_form(t, n, kappa, "<a'a>(t) vs n0 exp(-kappa t)")
    ok &= _rate_from_closed_form(t, np.abs(amp), kappa / 2.0, "|<a>(t)| vs |a0| exp(-kappa t / 2)")
    return ok


def _single_emitter(gdown=0.0, gphi=0.0):
    """One emitter, no cavity coupling, in its own rotating frame: pure dissipative dynamics."""
    M, Nc = 1, 2
    ops = build_ops(M, Nc)
    H = build_H(ops, 0.0, np.array([0.0]), np.array([0.0]))
    c = collapse_ops(ops, kappa=0.0, gdown=gdown, gphi=gphi)
    psi0 = qt.tensor(qt.basis(Nc, 0), (qt.basis(2, 0) + qt.basis(2, 1)).unit())   # equal superposition
    return ops, H, c, psi0


def gate4_emitter_decay():
    print("\n[C4] emitter decay: <sp sm> decays at gdown, |<sm>| at gdown/2")
    gdown = 0.23
    ops, H, c, psi0 = _single_emitter(gdown=gdown)
    t = np.linspace(0, 6, 13)
    pop = _decay_run(H, c, psi0, ops["ee"][0], t).real
    coh = _decay_run(H, c, psi0, ops["sm"][0], t)
    ok = _rate_from_closed_form(t, pop, gdown, "<sp sm>(t) vs p0 exp(-gdown t)")
    ok &= _rate_from_closed_form(t, np.abs(coh), gdown / 2.0, "|<sm>(t)| vs exp(-gdown t / 2)")
    return ok


def gate5_pure_dephasing():
    print("\n[C5] pure dephasing (gphi/2) D[sz]: |<sm>| decays at gphi, population untouched")
    gphi = 0.31
    ops, H, c, psi0 = _single_emitter(gphi=gphi)
    t = np.linspace(0, 6, 13)
    coh = _decay_run(H, c, psi0, ops["sm"][0], t)
    pop = _decay_run(H, c, psi0, ops["ee"][0], t).real
    ok = _rate_from_closed_form(t, np.abs(coh), gphi, "|<sm>(t)| vs exp(-gphi t)")
    ok &= _rate_from_closed_form(t, pop, 0.0, "<sp sm>(t) constant (dephasing moves no population)")
    return ok


def gate6_transverse_rate():
    print("\n[C6] combined transverse rate: |<sm>| decays at gamma_perp = gdown/2 + gphi")
    gdown, gphi = 0.19, 0.27
    gperp = gdown / 2.0 + gphi
    ops, H, c, psi0 = _single_emitter(gdown=gdown, gphi=gphi)
    t = np.linspace(0, 6, 13)
    coh = _decay_run(H, c, psi0, ops["sm"][0], t)
    ok = _rate_from_closed_form(
        t, np.abs(coh), gperp,
        f"|<sm>(t)| vs exp(-gamma_perp t), gamma_perp = {gperp:.3f}")
    return ok


def gate7_engine_carries_the_convention():
    """The same closed-form rates must come out of the CumulantEngine's own generator, so the
    convention is not merely right in the exact solver but identical inside the closure engine."""
    print("\n[C7] the CumulantEngine reproduces the same closed-form rates (g = 0, decoupled)")
    M, Nc = 1, 8
    kappa, gdown, gphi = 0.13, 0.19, 0.27
    eng = CumulantEngine(M, Nc, 0.0, np.array([0.0]), np.array([0.0]),
                         kappa=kappa, gdown=gdown, gphi=gphi, Fsafe=4)
    ops = eng.O
    ok = True
    # d<X>/dt = <Ldag[X]> evaluated on a state with known moments; compare to the analytic rate.
    psi = qt.tensor(qt.coherent(Nc, 0.5), (qt.basis(2, 0) + qt.basis(2, 1)).unit())
    rho = qt.ket2dm(psi)
    cases = [
        (ops["a"].dag() * ops["a"], kappa, "d<a'a>/dt = -kappa <a'a>"),
        (ops["a"], kappa / 2.0, "d<a>/dt = -(kappa/2) <a>"),
        (ops["ee"][0], gdown, "d<sp sm>/dt = -gdown <sp sm>"),
        (ops["sm"][0], gdown / 2.0 + gphi, "d<sm>/dt = -(gdown/2 + gphi) <sm>"),
    ]
    for O, rate, label in cases:
        lhs = complex(qt.expect(eng._Ldag(O), rho))
        rhs = -rate * complex(qt.expect(O, rho))
        err = abs(lhs - rhs)
        p = err < 1e-9
        ok = ok and p
        print(f"   {label:<52} |lhs - rhs| = {err:.3e}  {'PASS' if p else 'FAIL'}")
    return ok


def main():
    print("=" * 92)
    print("CONVENTION GATES: the model's dissipative conventions against closed-form results")
    print("=" * 92)
    gates = [
        ("C1 operator construction", gate1_scratch_operators),
        ("C2 sign conventions", gate2_sign_conventions),
        ("C3 cavity loss", gate3_cavity_loss),
        ("C4 emitter decay", gate4_emitter_decay),
        ("C5 pure dephasing", gate5_pure_dephasing),
        ("C6 transverse rate", gate6_transverse_rate),
        ("C7 engine consistency", gate7_engine_carries_the_convention),
    ]
    results = [(name, fn()) for name, fn in gates]
    print("\n" + "=" * 92)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}   {name}")
    allok = all(ok for _, ok in results)
    print("=" * 92)
    print(f"CONVENTION GATES: {'ALL PASS' if allok else 'FAILURE'}")
    print("gamma_perp = gdown/2 + gphi is a measured consequence of the model as coded, not an assumption.")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
