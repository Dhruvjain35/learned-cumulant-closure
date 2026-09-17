import os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path[:0]=['src/baselines','src/engines']
import numpy as np
from mps_lindblad import run_mps
from exact_tc import draw_disorder
WC,GDOWN,GC,KAPPA,GPHI,SIGMA = 1.0,0.02,0.7,0.03,0.02,0.15

def main():
    print("MPS FEASIBILITY at sigma=0.15 (POCKET): can the tensor-network reference reach M>=12?")
    print("The question is whether the bond dimension saturates or explodes through the burst.\n",flush=True)
    print(f"{'M':>3}{'Nc':>4}{'chi_max':>8} | {'chi reached':>12}{'n_peak':>9}{'Tr drift':>10}{'secs':>9}{'  verdict':>10}")
    CELLS=[(4,10,256),(6,14,256),(8,16,256),(10,20,256),(12,22,256)]
    for M,Nc,chi in CELLS:
        w=draw_disorder(M,WC,SIGMA,seed=7,family="gauss"); g=np.full(M,GC/np.sqrt(M))
        t0=time.time()
        try:
            r=run_mps(M,Nc,WC,w,g,KAPPA,GDOWN,GPHI,tmax=10.0,nt=51,chi_max=chi)
            sat = "SATURATED" if r["chi_max_reached"]>=chi else "ok"
            print(f"{M:>3}{Nc:>4}{chi:>8} | {r['chi_max_reached']:>12}{max(r['n']):>9.4f}"
                  f"{abs(np.array(r['trace'])-1).max():>10.1e}{r['seconds']:>9.1f}{sat:>10}",flush=True)
        except Exception as e:
            print(f"{M:>3}{Nc:>4}{chi:>8} | FAILED after {time.time()-t0:.0f}s: {type(e).__name__} {str(e)[:70]}",flush=True)
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
