import sys; sys.path[:0]=['src/evaluation','src/closures','src/engines','src']
import numpy as np
from closure_v3 import ClosureApplyV3, ratio_transform
from closure_v6 import ClosureApplyV6, run_v6, lambdas
from pilot_eval import build_cell, m0_product, run_from, WC, TMAX, NT
from v7 import route, _load
GC=0.7

def main():
    print("Is the burst peak reached BEFORE the physicality violation? (POCKET, sigma=0)\n")
    print(f"{'M':>4}{'branch':>7} | {'t_peak':>8}{'t_first_violation':>19}{'t_maxabs>100':>14}{'peak value':>12}")
    for M in (10,12,16):
        ka,gp=0.03,0.02
        eng,w,g=build_cell(M,GC,ka,gp,0.0,7); det=w-WC; regime=(GC,ka,gp); m0=m0_product(eng,1.0)
        _,tr,_=run_v6(eng,m0,ClosureApplyV6(_load("v6"),eng,det,regime,xform=ratio_transform),lambdas(ka,gp))
        t=np.linspace(0,TMAX,NT)
        n=np.array([r[eng.tpos[(1,1,tuple([0]*M))]].real for r in tr])
        cols=[eng.tpos[(0,0,tuple(3 if s==i else 0 for s in range(M)))] for i in range(M)]
        z=np.array([[r[c].real for c in cols] for r in tr])
        bad=(z.min(axis=1)<-1-1e-9)|(z.max(axis=1)>1+1e-9)|(n<-1e-9)
        big=np.max(np.abs(np.array(tr)),axis=1)>100
        tv=t[np.argmax(bad)] if bad.any() else np.nan
        tb=t[np.argmax(big)] if big.any() else np.nan
        k=int(np.argmax(n))
        print(f"{M:>4}{'v6':>7} | {t[k]:>8.2f}{tv:>19.2f}{tb:>14.2f}{n[k]:>12.4f}",flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
