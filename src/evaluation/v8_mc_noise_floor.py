import sys, time; sys.path[:0]=['src/engines','src/evaluation','src']
import numpy as np, qutip as qt
from exact_tc import build_ops, build_H, collapse_ops, draw_disorder
WC,GDOWN,GC,SIGMA = 1.0,0.02,0.7,0.15

def main():
    print("Monte-Carlo noise floor at M=12, sigma=0.15: can trajectories discriminate a ~5% V7 error?")
    print("If mcerr/n_peak is not well under V7's error, the comparison is useless no matter how long we run.\n",flush=True)
    print(f"{'M':>3}{'Nc':>4}{'ntraj':>7} | {'n_peak':>9}{'mcerr':>9}{'mcerr/peak':>12}{'secs':>9}{'  ntraj for 1%':>15}")
    for M,Nc,ntraj,ka,gp in [(10,20,60,0.03,0.02),(12,22,60,0.03,0.02)]:
        w=draw_disorder(M,WC,SIGMA,50,"gauss"); g=np.full(M,GC/np.sqrt(M))
        ops=build_ops(M,Nc); H=build_H(ops,WC,w,g); c=collapse_ops(ops,kappa=ka,gdown=GDOWN,gphi=gp)
        psi0=qt.tensor(qt.basis(Nc,0),*[qt.basis(2,0)]*M)
        ts=np.linspace(0,10.0,51); nop=ops["a"].dag()*ops["a"]
        t0=time.time()
        r=qt.mcsolve(H,psi0,ts,c_ops=c,e_ops=[nop],ntraj=ntraj,options={"progress_bar":False,"map":"serial"})
        el=time.time()-t0
        n=np.array(r.expect[0]).real; k=int(np.argmax(n))
        se=np.array(r.std_expect[0]).real/np.sqrt(ntraj)
        need=int(np.ceil(ntraj*(se[k]/(0.01*n[k]))**2))
        print(f"{M:>3}{Nc:>4}{ntraj:>7} | {n[k]:>9.4f}{se[k]:>9.4f}{100*se[k]/n[k]:>11.2f}%{el:>9.1f}{need:>15}",flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
