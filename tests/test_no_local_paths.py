"""
Packaging test: no tracked source file may hard-code a user-specific absolute path (release portability).
Frozen result logs under results/ are artifacts and are exempt (they record where they were produced).
Run:  python run.py tests/test_no_local_paths.py
"""
import re
import subprocess
import sys

PAT = re.compile(r"(/Users/[A-Za-z0-9_.-]+/|/home/[A-Za-z0-9_.-]+/|C:\\\\Users\\\\)")
SCAN_EXT = (".py", ".jl", ".toml", ".tex", ".cff", "Makefile", ".cfg", ".yml", ".yaml", ".sh")


def main():
    files = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout.split()
    bad = []
    for f in files:
        if f.startswith(("results/", "external_validation/")) or not f.endswith(SCAN_EXT):
            continue
        for i, line in enumerate(open(f, encoding="utf-8", errors="ignore"), 1):
            if PAT.search(line):
                bad.append(f"{f}:{i}: {line.strip()[:100]}")
    if bad:
        print("hard-coded local paths found:\n" + "\n".join(bad))
        return 1
    print(f"no hard-coded local paths in {sum(1 for f in files if f.endswith(SCAN_EXT))} tracked source files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
