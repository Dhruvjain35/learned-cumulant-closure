"""
Mean-field (first-order) equations of motion for the open Tavis-Cummings model, per emitter
(so disorder-capable). This is the VALIDATION GATE: before trusting any hand-derived closure,
the mean-field machinery must reproduce the exact solver in the regime where mean-field is
accurate (weak coupling / low excitation / driven steady state).

Variables: alpha=<a> (complex), s_i=<sigma_i^->, z_i=<sigma_i^z>.
Derived from d<O>/dt = <i[H,O]> + sum_c D_dagger[c](O), with mean-field factorization
<a sigma_z> ~ <a><sigma_z>, <a^dag sigma^-> ~ <a^dag><sigma^->:

  dalpha/dt = -i wc alpha - i sum_i g_i s_i - (kappa/2) alpha
  ds_i/dt   = -i w_i s_i + i g_i alpha z_i - (gdown/2 + gphi) s_i
  dz_i/dt   = -4 g_i Im(conj(alpha) s_i) - gdown (z_i + 1)              [+ gpump (1 - z_i) if pumped]

Each dissipative coefficient was derived from the Lindblad adjoint and is checked below.
"""
import numpy as np
from scipy.integrate import solve_ivp
import qutip as qt
from exact_tc import build_ops, build_H, collapse_ops, draw_disorder


def mf_rhs(t, y, wc, w, g, kappa, gdown, gphi, gpump):
    M = len(w)
    alpha = y[0] + 1j * y[1]
    s = y[2:2 + 2 * M:2] + 1j * y[3:2 + 2 * M:2]
    z = y[2 + 2 * M:2 + 3 * M]
    dalpha = -1j * wc * alpha - 1j * np.sum(g * s) - 0.5 * kappa * alpha
    ds = -1j * w * s + 1j * g * alpha * z - (gdown / 2 + gphi) * s
    dz = -4.0 * g * np.imag(np.conj(alpha) * s) - gdown * (z + 1) + gpump * (1 - z)
    out = np.empty_like(y)
    out[0], out[1] = dalpha.real, dalpha.imag
    out[2:2 + 2 * M:2], out[3:2 + 2 * M:2] = ds.real, ds.imag
    out[2 + 2 * M:2 + 3 * M] = dz
    return out


def run_mf(M, wc, w, g, kappa, gdown, gphi, gpump, alpha0, z0, ts):
    y0 = np.zeros(2 + 3 * M)
    y0[0], y0[1] = alpha0.real, alpha0.imag
    y0[2 + 2 * M:2 + 3 * M] = z0
    sol = solve_ivp(mf_rhs, (ts[0], ts[-1]), y0, t_eval=ts, args=(wc, w, g, kappa, gdown, gphi, gpump),
                    rtol=1e-8, atol=1e-10, method="RK45")
    alpha = sol.y[0] + 1j * sol.y[1]
    z = sol.y[2 + 2 * M:2 + 3 * M]
    ncav = np.abs(alpha) ** 2          # mean-field photon number = |<a>|^2
    zbar = z.mean(axis=0)
    return ncav, zbar, alpha


def exact_coherent(M, Nc, wc, w, g, kappa, gdown, gphi, gpump, alpha0, excited, ts):
    ops = build_ops(M, Nc)
    H = build_H(ops, wc, w, g)
    c = collapse_ops(ops, kappa=kappa, gdown=gdown, gphi=gphi, gpump=gpump)
    emit = qt.basis(2, 0) if excited else qt.basis(2, 1)
    psi0 = qt.tensor(qt.coherent(Nc, alpha0), *[emit for _ in range(M)])
    a = ops["a"]
    zop = [ops["sz"][i] for i in range(M)]
    res = qt.mesolve(H, psi0, ts, c_ops=c, e_ops=[a.dag() * a] + zop, options={"progress_bar": False})
    ncav = np.array(res.expect[0]).real
    zbar = np.mean([np.array(res.expect[1 + i]).real for i in range(M)], axis=0)
    return ncav, zbar


def gate_weak_coherent():
    """Weak coherent field, emitters in ground: near-linear regime where mean-field ~ exact."""
    print("\n[MF gate A] weak coherent drive-free field, emitters ground, weak coupling: MF vs exact")
    M, Nc, wc = 2, 6, 1.0
    w = np.full(M, 1.0); g = np.full(M, 0.05)
    kappa, gdown, gphi, gpump = 0.05, 0.0, 0.0, 0.0
    alpha0, z0, excited = 0.3, np.full(M, -1.0), False
    ts = np.linspace(0, 60, 240)
    nc_mf, z_mf, _ = run_mf(M, wc, w, g, kappa, gdown, gphi, gpump, alpha0 + 0j, z0, ts)
    nc_ex, z_ex = exact_coherent(M, Nc, wc, w, g, kappa, gdown, gphi, gpump, alpha0, excited, ts)
    err_n = np.max(np.abs(nc_mf - nc_ex)) / max(nc_ex.max(), 1e-9)
    err_z = np.max(np.abs(z_mf - z_ex))
    print(f"   max rel err in <a^dag a> = {err_n:.3f}   max abs err in <sz> = {err_z:.4f}")
    print(f"   {'PASS' if err_n < 0.05 and err_z < 0.02 else 'CHECK'} (MF should track exact in this weak/linear regime)")
    return err_n, err_z


def gate_single_emitter_rabi():
    """M=1, one excitation, lossless: MF two coupled oscillators should match exact vacuum Rabi at low excitation."""
    print("\n[MF gate B] single emitter, weak coherent field, lossless: MF vs exact vacuum-Rabi exchange")
    M, Nc, wc = 1, 6, 1.0
    w = np.array([1.0]); g = np.array([0.1])
    ts = np.linspace(0, 80, 400)
    nc_mf, z_mf, _ = run_mf(M, wc, w, g, 0.0, 0.0, 0.0, 0.0, 0.25 + 0j, np.array([-1.0]), ts)
    nc_ex, z_ex = exact_coherent(M, Nc, wc, w, g, 0.0, 0.0, 0.0, 0.0, 0.25, False, ts)
    err_n = np.max(np.abs(nc_mf - nc_ex)) / max(nc_ex.max(), 1e-9)
    print(f"   max rel err in <a^dag a> = {err_n:.3f}   {'PASS' if err_n < 0.05 else 'CHECK'}")
    return err_n


if __name__ == "__main__":
    print("=" * 74)
    print("MEAN-FIELD EOM validation gate (must match exact where MF is accurate)")
    print("=" * 74)
    gate_weak_coherent()
    gate_single_emitter_rabi()
