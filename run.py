#!/usr/bin/env python3
"""
Run any module in src/ with the project's flat-import path shim, from the repository root.

    python run.py src/analysis/mechanism.py [args...]

The code base uses flat imports (`from exact_tc import ...`) and reads/writes hard-coded `results/` paths.
This runner puts every `src/*` subpackage on `sys.path` so those flat imports resolve regardless of which
subpackage a module lives in, and it runs from the repo root so `results/` resolves correctly. This keeps the
(frozen, validated) code unchanged. Always invoke from the repository root.
"""
import sys, os, glob, runpy

ROOT = os.path.dirname(os.path.abspath(__file__))
for d in sorted(glob.glob(os.path.join(ROOT, "src", "*"))):
    if os.path.isdir(d):
        sys.path.insert(0, d)

if len(sys.argv) < 2:
    print("usage: python run.py <path/to/script.py> [args...]", file=sys.stderr)
    raise SystemExit(2)

script = sys.argv[1]
if not os.path.isfile(script):
    print(f"no such script: {script}", file=sys.stderr)
    raise SystemExit(2)

sys.argv = sys.argv[1:]                     # hand the remaining args to the target script
runpy.run_path(script, run_name="__main__")
