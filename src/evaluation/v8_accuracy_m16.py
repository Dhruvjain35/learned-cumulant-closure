import sys, time; sys.path[:0]=['src/evaluation','src/closures','src/engines','src/baselines','src']
import numpy as np
from v8_accuracy_ladder import closure_peak, FAC
from piqs_reference import burst_piqs
from v7 import route
GC, GDOWN = 0.7, 0.02

def main():
    print("Extending the accuracy ladder to M=16 (sigma=0): is the error growth LINEAR or SATURATING?")
    print("This decides whether V7's advantage has a crossover at reachable M.\n", flush=True)
    print(f"{'regime':<9}{'M':>4} | {'exact n_peak':>13}{'V7 relerr':>11}{'plain relerr':>13}{'ref secs':>10}")
    for rg, ka, gp in [("CONTROL",0.15,0.02), ("POCKET",0.03,0.02)]:
        t0=time.time()
        _,_,fpk,_ = burst_piqs(16, 36, 1.0, GC, ka, GDOWN, gp, tmax=40.0, nt=200, rate_factors=FAC)
        npk = fpk/ka; el=time.time()-t0
        v7 = closure_peak(16, ka, gp, route(ka,gp)); pl = closure_peak(16, ka, gp, "plain")
        f=lambda v: f"{(v-npk)/npk:+.4f}" if v is not None else "DIVERGED"
        print(f"{rg:<9}{16:>4} | {npk:>13.5f}{f(v7):>11}{f(pl):>13}{el:>10.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
