"""
Import smoke test: every subpackage's load-bearing modules import cleanly through the run.py path shim.
This does NOT run any simulation (module import does not trigger a script's `__main__` block) — it only
confirms the reorganized folder layout did not break the project's flat imports.

Run from the repository root:  python run.py tests/test_imports.py
"""
import importlib

MODULES = [
    # engines
    "exact_tc", "cumulant", "scalable_engine", "mean_field", "mc_ref", "baseline_shell",
    # closures
    "local_closure", "closure_v3", "closure_v6", "v7",
    # baselines
    "t3_truncation", "t4_truncation", "baselines_eval", "t4_eval",
    # evaluation
    "pilot_eval", "burst_pred", "burst_t3", "eval_v7",
    # analysis
    "mechanism", "mechanism_report", "disorder_sweep", "step1_gc_sweep",
]

if __name__ == "__main__":
    failed = []
    for m in MODULES:
        try:
            importlib.import_module(m)
            print("ok   ", m)
        except Exception as e:
            failed.append((m, repr(e)))
            print("FAIL ", m, "->", repr(e))
    print(f"\n{len(MODULES) - len(failed)}/{len(MODULES)} modules imported cleanly")
    if failed:
        raise SystemExit(1)
