# Learned Cumulant Closure — convenience targets. Always run from the repository root.
# Python entry points go through run.py (a flat-import path shim so the reorganized layout keeps working);
# the Julia reproduction is standalone.
PY ?= python

.PHONY: help test gates burst mechanism fourth-order replicate portable-export portable-test figures readme-figures quantum-figures render demo-data demo publish-demo result40 result40-smoke result42 result42-regression

help:
	@echo "make test          # import smoke test (fast, no simulation)"
	@echo "make gates         # analytic limits, model conventions, and the EOM identity (~3 min)"
	@echo "make burst         # exact vs 2nd-order vs V7 superradiant burst-peak comparison"
	@echo "make mechanism     # connected-kappa3 channel attribution + closed-loop ablation"
	@echo "make fourth-order  # exact vs 2nd vs 3rd vs 4th vs V7 fairness test"
	@echo "make replicate     # Julia cross-platform reproduction (needs Julia; see docs/REPLICATION_QUICKSTART.md)"
	@echo "make portable-export # re-export frozen V7 to portable_v7/ (NPZ+JSON; byte-identical on rerun)"
	@echo "make portable-test   # portable V7 vs frozen pickles: features, networks, kappa3, full rollouts (~1-2 min)"
	@echo "make figures        # render the 5 main manuscript figures from committed results (no simulation)"
	@echo "make readme-figures # render the README figures (light + dark) from committed results"
	@echo "make demo-data      # re-run the 30 held-out cells for the demo and check them against the record (~6 min, 8 cores)"
	@echo "make quantum-figures # exact P(n,t) and Wigner functions for held-out cell H11 (~10 s)"
	@echo "make render          # render the cavity animation from the exact solution (~1 min)"
	@echo "make demo           # serve demo/ at http://127.0.0.1:8000"
	@echo "make publish-demo   # push demo/ to the gh-pages branch (GitHub Pages)"
	@echo "make result40       # re-run Result 40 (held-out parameters, M=5, ~50 min); rewrites results/heldout_param_*"
	@echo "make result42       # re-run Result 42 (M=5 mechanism, ~15 min); rewrites results/mechanism_m5*"
	@echo ""
	@echo "run any module:    $(PY) run.py src/<subpackage>/<module>.py"

test:
	$(PY) run.py tests/test_imports.py
	$(PY) run.py tests/test_no_local_paths.py
# Every derivation check, all deterministic pass/fail, all exiting non-zero on failure.
# Runs in CI on every commit. About three minutes total.
gates:
	$(PY) run.py src/validation/validate_model.py
	$(PY) run.py src/validation/check_conventions.py
	$(PY) run.py src/validation/check_eom_identity.py
burst:
	$(PY) run.py src/evaluation/burst_pred.py
mechanism:
	$(PY) run.py src/analysis/mechanism.py
fourth-order:
	$(PY) run.py src/baselines/t4_eval.py
replicate:
	julia --project=. julia_setup.jl && julia --project=. replicate_cumulant.jl
portable-export:
	$(PY) run.py src/tools/export_portable_v7.py
portable-test:
	$(PY) run.py tests/test_portable_v7.py
# Render-only: reads committed results/ artifacts, recomputes no science. Output: figures/fig1..fig5 (.pdf, .png).
figures:
	$(PY) run.py scripts/build_manuscript_figures.py
readme-figures:
	$(PY) run.py scripts/make_readme_figures.py
demo-data:
	$(PY) run.py scripts/export_demo_static.py
	$(PY) run.py scripts/export_demo_data.py --workers 8
quantum-figures:
	$(PY) run.py scripts/export_quantum_states.py
	$(PY) run.py scripts/make_quantum_figures.py
render:
	$(PY) run.py scripts/render_cavity.py
demo:
	cd demo && $(PY) -m http.server 8000 --bind 127.0.0.1
publish-demo:
	git push -f origin $$(git subtree split --prefix demo HEAD):refs/heads/gh-pages
# Re-run the pre-registered campaigns (science; deterministic). They write to results/ -- afterwards
# `git diff --stat results/` shows whether the committed artifacts were reproduced exactly.
result40-smoke:
	$(PY) run.py src/evaluation/heldout_parameter_burst.py --smoke
result40:
	@echo "Re-running Result 40 (~30 min). Overwrites results/heldout_param_*; compare with git diff afterwards."
	$(PY) run.py src/evaluation/heldout_parameter_burst.py
result42-regression:
	$(PY) run.py src/analysis/mechanism_m5.py --regression
result42:
	@echo "Re-running Result 42 (~15 min). Overwrites results/mechanism_m5*; compare with git diff afterwards."
	$(PY) run.py src/analysis/mechanism_m5.py --campaign
