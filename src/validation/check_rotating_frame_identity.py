"""
POST-HOC companion to check_carrier_invariance.py. NOT pre-registered: written after the P2 audit's result
was known, to pin down two facts its diagnostic pointed at. It decides nothing in the P2 verdict.

Part A -- the frozen runs ARE rotating-frame runs. Every frozen result uses omega_c = 1 in the lab frame.
Re-run the four P2 cells (POCKET/CONTROL, M = 2, 3, draw 50) with omega_c = 0 and w_i = Delta_i (the frame
rotating at the old carrier, i.e. C = -1) at the FROZEN tolerances, and compare exact, plain and V7 n(t) with
the omega_c = 1 runs.

Part B -- where the P2 closure discrepancies come from. Plain second order, POCKET M = 2: the largest charged
tracked moment and the n(t) difference from omega_c = 1, for omega_c in {1, 0, 10, 100}, at the frozen
(rtol 1e-4 / atol 1e-6) and a tight (1e-8 / 1e-10) RK45 tolerance. Also the largest EOM coefficient that
couples moments of different excitation-number charge (exactly zero in the true equations; the least-squares
projection leaves rounding residue).

    python run.py src/validation/check_rotating_frame_identity.py
"""
import numpy as np

from check_carrier_invariance import build_eng, charge, eng_m0, rel_traj, exact_run, REGIMES
from pilot_eval import run_from
from local_run import observables
from v7 import run_v7


def main():
    L = []
    say = lambda s="": (print(s, flush=True), L.append(s))
    say("ROTATING-FRAME IDENTITY (POST-HOC, not pre-registered; companion to the P2 carrier audit)")
    say("\nPart A: omega_c = 0 (rotating frame) vs the frozen omega_c = 1, frozen tolerances, draw 50")
    for name, reg in REGIMES.items():
        for M in (2, 3):
            e1, e0 = exact_run(M, *reg, 0.0), exact_run(M, *reg, -1.0)
            r = {}
            for C in (0.0, -1.0):
                eng, det = build_eng(M, *reg, C)
                m0 = eng_m0(eng)
                tp = run_from(eng, m0, None)[1]
                _, t7 = run_v7(eng, m0, det, reg)
                Q = np.array([charge(k) for k in eng.track])
                r[C] = (observables(eng, tp)[0], observables(eng, t7)[0], float(np.abs(t7[:, Q != 0]).max()))
            say(f"  {name:<7s} M={M}  exact n rel diff {rel_traj(e0['n'], e1['n']):.1e}   "
                f"plain n {rel_traj(r[-1.0][0], r[0.0][0]):.1e}   V7 n {rel_traj(r[-1.0][1], r[0.0][1]):.1e}   "
                f"V7 charged max (omega_c=0) {r[-1.0][2]:.1e}")

    say("\nPart B: plain second order, POCKET M=2 -- charged-moment growth vs carrier and tolerance")
    eng, _ = build_eng(2, *REGIMES["POCKET"], 0.0)
    Q = np.array([charge(k) for k in eng.track])
    T2, T3 = eng.T2.toarray(), eng.T3.toarray()
    Q3 = np.array([charge(k) for k in eng.o3_keys])
    v2 = max(abs(T2[a, b]) for a in range(len(Q)) for b in range(len(Q)) if Q[a] != Q[b])
    v3 = max(abs(T3[a, c]) for a in range(len(Q)) for c in range(len(Q3)) if Q[a] != Q3[c])
    say(f"  largest charge-violating EOM coefficient: T2 {v2:.1e}   T3 {v3:.1e}   (exact value 0)")
    for rtol, atol in ((1e-4, 1e-6), (1e-8, 1e-10)):
        base = None
        for C in (0.0, -1.0, 9.0, 99.0):
            e, _ = build_eng(2, *REGIMES["POCKET"], C)
            tr = run_from(e, eng_m0(e), None, rtol=rtol, atol=atol)[1]
            n = observables(e, tr)[0]
            base = n if C == 0.0 else base
            say(f"  rtol {rtol:.0e}  omega_c={1 + C:>5.0f}  charged max {np.abs(tr[:, Q != 0]).max():.1e}   "
                f"n rel diff vs omega_c=1 {rel_traj(n, base):.1e}")
    with open("results/rotating_frame_identity.log", "w") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
