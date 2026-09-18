import sys, time; sys.path[:0]=['src/evaluation','src/closures','src/engines','src']
import numpy as np, qutip as qt
from closure_v3 import ClosureApplyV3, ratio_transform
from exact_tc import build_ops, build_H, collapse_ops, draw_disorder
from pilot_eval import build_cell, m0_product, run_from, WC
from v7 import _load
GDOWN,GC,SIGMA = 0.02,0.7,0.15
NTRAJ,TMAX,NT = 400,10.0,51
def cp(M,ka,gp,which,seed):
    eng,w,g=build_cell(M,GC,ka,gp,SIGMA,seed); det=w-WC; m0=m0_product(eng,1.0)
    if which=="plain": _,tr=run_from(eng,m0,None)
    else: _,tr=run_from(eng,m0,ClosureApplyV3(_load("v3dens"),eng,det,(GC,ka,gp),xform=ratio_transform))
    if not np.isfinite(tr).all(): return None
    return float(np.max([r[eng.tpos[(1,1,tuple([0]*M))]].real for r in tr]))

def main():
    print("Is CONTROL's scatter Monte-Carlo noise or DISORDER-REALIZATION scatter?")
    print("Same cell, three seeds. MC error is ~1%; if plain moves several points, it is the realization.\n",flush=True)
    print(f"{'seed':>5}{'M':>4} | {'n_peak(mc)':>11}{'+/-%':>7}{'V7':>10}{'plain':>10}{'secs':>8}")
    for seed in (50,51,52):
        M,Nc,ka,gp = 10,20,0.15,0.02
        w=draw_disorder(M,WC,SIGMA,seed,"gauss"); g=np.full(M,GC/np.sqrt(M))
        ops=build_ops(M,Nc); H=build_H(ops,WC,w,g); c=collapse_ops(ops,kappa=ka,gdown=GDOWN,gphi=gp)
        psi0=qt.tensor(qt.basis(Nc,0),*[qt.basis(2,0)]*M); ts=np.linspace(0,TMAX,NT)
        t0=time.time()
        r=qt.mcsolve(H,psi0,ts,c_ops=c,e_ops=[ops["a"].dag()*ops["a"]],ntraj=NTRAJ,options={"progress_bar":False})
        n=np.array(r.expect[0]).real; k=int(np.argmax(n)); ref=float(n[k])
        se=float(np.array(r.std_expect[0]).real[k]/np.sqrt(NTRAJ))
        v7=cp(M,ka,gp,"v4",seed); pl=cp(M,ka,gp,"plain",seed)
        f=lambda v: f"{(v-ref)/ref:+.4f}" if v is not None else "DIV"
        print(f"{seed:>5}{M:>4} | {ref:>11.4f}{100*se/ref:>6.2f}%{f(v7):>10}{f(pl):>10}{time.time()-t0:>8.1f}",flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
