import sys, time; sys.path[:0]=['src/evaluation','src/closures','src/engines','src']
import numpy as np, qutip as qt
from closure_v3 import ClosureApplyV3, ratio_transform
from closure_v6 import ClosureApplyV6, run_v6, lambdas
from exact_tc import build_ops, build_H, collapse_ops, draw_disorder
from pilot_eval import build_cell, m0_product, run_from, WC
from v7 import route, _load
GDOWN,GC,SIGMA,SEED = 0.02,0.7,0.15,50
NTRAJ, TMAX, NT = 400, 10.0, 51

def closure_peak(M, ka, gp, which, sigma, seed):
    eng,w,g = build_cell(M,GC,ka,gp,sigma,seed)
    det=w-WC; regime=(GC,ka,gp); m0=m0_product(eng,1.0)
    if which=="plain": _,tr = run_from(eng,m0,None)
    elif which=="v6":  _,tr,_ = run_v6(eng,m0,ClosureApplyV6(_load("v6"),eng,det,regime,xform=ratio_transform),lambdas(ka,gp))
    else:              _,tr = run_from(eng,m0,ClosureApplyV3(_load("v3dens"),eng,det,regime,xform=ratio_transform))
    if not np.isfinite(tr).all(): return None
    col=eng.tpos[(1,1,tuple([0]*M))]
    return float(np.max([r[col].real for r in tr]))


def main():
    print("DISORDERED ladder M=8,10 by quantum trajectories (ntraj=400, t<=10).")
    print("Tests V7 in the regime it was BUILT for (sigma>0) at the size where it diverges.\n",flush=True)
    print(f"{'regime':<9}{'M':>3} | {'n_peak (mc)':>12}{'+/-%':>7} | {'V7 relerr':>11}{'plain relerr':>13}{'secs':>8}")
    for rg,ka,gp in [("POCKET",0.03,0.02),("CONTROL",0.15,0.02)]:
      for M,Nc in [(8,18),(10,20)]:
          w=draw_disorder(M,WC,SIGMA,SEED,"gauss"); g=np.full(M,GC/np.sqrt(M))
          ops=build_ops(M,Nc); H=build_H(ops,WC,w,g); c=collapse_ops(ops,kappa=ka,gdown=GDOWN,gphi=gp)
          psi0=qt.tensor(qt.basis(Nc,0),*[qt.basis(2,0)]*M); ts=np.linspace(0,TMAX,NT)
          t0=time.time()
          r=qt.mcsolve(H,psi0,ts,c_ops=c,e_ops=[ops["a"].dag()*ops["a"]],ntraj=NTRAJ,options={"progress_bar":False})
          el=time.time()-t0
          n=np.array(r.expect[0]).real; k=int(np.argmax(n)); ref=float(n[k])
          se=float(np.array(r.std_expect[0]).real[k]/np.sqrt(NTRAJ))
          v7=closure_peak(M,ka,gp,route(ka,gp),SIGMA,SEED); pl=closure_peak(M,ka,gp,"plain",SIGMA,SEED)
          f=lambda v: f"{(v-ref)/ref:+.4f}" if v is not None else "DIVERGED"
          print(f"{rg:<9}{M:>3} | {ref:>12.4f}{100*se/ref:>6.2f}% | {f(v7):>11}{f(pl):>13}{el:>8.1f}",flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
