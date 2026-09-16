import sys; sys.path[:0]=['src/evaluation','src/closures','src/engines','src/analysis','src']
import numpy as np
from closure_v3 import ClosureApplyV3, ratio_transform
from closure_v6 import ClosureApplyV6, run_v6, lambdas
from pilot_eval import build_cell, m0_product, run_from, WC
from v7 import _load

GC=0.7
def per_site_z(eng,tr):
    cols=[eng.tpos[(0,0,tuple(3 if s==i else 0 for s in range(eng.M)))] for i in range(eng.M)]
    return np.array([[r[c].real for c in cols] for r in tr])
def photon(eng,tr): return np.array([r[eng.tpos[(1,1,tuple([0]*eng.M))]].real for r in tr])

def run(M, ka, gp, branch, sigma):
    eng,w,g = build_cell(M,GC,ka,gp,sigma,7)
    det=w-WC; regime=(GC,ka,gp); m0=m0_product(eng,1.0)
    if branch=="v6":
        _,tr,_ = run_v6(eng,m0,ClosureApplyV6(_load("v6"),eng,det,regime,xform=ratio_transform),lambdas(ka,gp))
    else:
        _,tr = run_from(eng,m0,ClosureApplyV3(_load("v3dens"),eng,det,regime,xform=ratio_transform))
    if not np.isfinite(tr).all(): return dict(phys=False,zmax=np.inf,nmin=-np.inf,mx=np.inf)
    z,n=per_site_z(eng,tr),photon(eng,tr)
    return dict(phys=bool(z.min()>=-1-1e-9 and z.max()<=1+1e-9 and n.min()>=-1e-9),
                zmax=float(z.max()),nmin=float(n.min()),mx=float(np.max(np.abs(tr))))

print("DECISIVE TEST: separate BRANCH from REGIME. POCKET regime (kappa=0.03) forced to BOTH branches,")
print("at sigma=0 and sigma=0.15. If V4 is clean at sigma=0 but fails at sigma=0.15 in the SAME regime,")
print("the two branches fail for different reasons.\n")
print(f"{'regime':<9}{'branch':>7}{'sigma':>7}{'M':>4} | {'phys':>6}{'z_max':>13}{'n_min':>13}{'max|m|':>13}")
for ka,gp,rg in [(0.03,0.02,'POCKET')]:
    for sigma in (0.0, 0.15):
        for branch in ('v4','v6'):
            for M in (10,12,16):
                r=run(M,ka,gp,branch,sigma)
                print(f"{rg:<9}{branch:>7}{sigma:>7}{M:>4} | {str(r['phys']):>6}{r['zmax']:>13.4g}{r['nmin']:>13.4g}{r['mx']:>13.4g}",flush=True)
