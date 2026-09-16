import sys, time; sys.path[:0]=['src/evaluation','src/closures','src/engines','src/baselines','src']
import numpy as np
from closure_v3 import ClosureApplyV3, ratio_transform
from closure_v6 import ClosureApplyV6, run_v6, lambdas
from pilot_eval import build_cell, m0_product, run_from, WC
from v7 import route, _load
from piqs_reference import burst_piqs, calibrate_piqs_rates

GC, GDOWN = 0.7, 0.02
FAC = calibrate_piqs_rates(verbose=False)
NCMAP = {5:16, 8:20, 10:24, 12:28, 16:36}

def closure_peak(M, ka, gp, which):
    eng,w,g = build_cell(M,GC,ka,gp,0.0,7)
    det=w-WC; regime=(GC,ka,gp); m0=m0_product(eng,1.0)
    if which=="plain":
        _,tr = run_from(eng,m0,None)
    elif which=="v6":
        _,tr,_ = run_v6(eng,m0,ClosureApplyV6(_load("v6"),eng,det,regime,xform=ratio_transform),lambdas(ka,gp))
    else:
        _,tr = run_from(eng,m0,ClosureApplyV3(_load("v3dens"),eng,det,regime,xform=ratio_transform))
    if not np.isfinite(tr).all(): return None
    n = np.array([r[eng.tpos[(1,1,tuple([0]*M))]].real for r in tr])
    return float(n.max())


def main():
    print("V8 ACCURACY LADDER at sigma=0: closures vs the EXACT permutation-invariant reference.")
    print("This is the first time any closure in this project has been checked above M=10.\n")
    print(f"{'regime':<9}{'M':>4}{'branch':>8} | {'n_peak exact':>13}{'V7 relerr':>12}{'plain relerr':>14}{'ref secs':>10}")
    for rg, ka, gp in [("CONTROL",0.15,0.02), ("POCKET",0.03,0.02)]:
        for M in (5,8,10,12):
            t0=time.time()
            _,_,fpk,_ = burst_piqs(M, NCMAP[M], 1.0, GC, ka, GDOWN, gp, tmax=40.0, nt=200, rate_factors=FAC)
            npk_exact = fpk/ka
            el=time.time()-t0
            br = route(ka,gp)
            pv7 = closure_peak(M,ka,gp,br)
            ppl = closure_peak(M,ka,gp,"plain")
            f=lambda v: f"{(v-npk_exact)/npk_exact:+.4f}" if v is not None else "DIVERGED"
            print(f"{rg:<9}{M:>4}{br:>8} | {npk_exact:>13.5f}{f(pv7):>12}{f(ppl):>14}{el:>10.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
