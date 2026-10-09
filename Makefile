# Learned Cumulant Closure — convenience targets. Always run from the repository root.
# Python entry points go through run.py (a flat-import path shim so the reorganized layout keeps working);
# the Julia reproduction is standalone.
PY ?= python

.PHONY: help test gates burst mechanism fourth-order replicate portable-export portable-test figures paper supplement manuscript-check result40 result40-smoke result42 result42-regression release-preview arxiv-preview

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
	@echo "make paper          # compile the PRA REVTeX manuscript paper/main.pdf (render only; needs tectonic)"
	@echo "make supplement     # render SM tables from committed results and compile paper/supplement.pdf"
	@echo "make manuscript-check # numbers, audited claims, citations, locked wording and style of the manuscript"
	@echo "make result40       # re-run Result 40 (held-out parameters, M=5, ~30 min); rewrites results/heldout_param_*"
	@echo "make result42       # re-run Result 42 (M=5 mechanism, ~15 min); rewrites results/mechanism_m5*"
	@echo "make release-preview# stage the planned public release under release/v1.0.0-preview (local only)"
	@echo "make arxiv-preview  # assemble and test-compile an arXiv source bundle under release/arxiv-preview"
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
# PRA submission source (paper/). Render only: no scientific computation. TeX engine: tectonic (fetches TeX packages
# on first use); any REVTeX 4.2 installation with pdflatex+bibtex also works (see paper/README.md).
TECTONIC ?= tectonic
paper:
	python3 paper/tools/check_bib.py
	cd paper && $(TECTONIC) -X compile --keep-logs main.tex
supplement:
	$(PY) run.py paper/tools/build_supplement_tables.py
	cd paper && $(TECTONIC) -X compile --keep-logs supplement.tex
# Prose-integrity check of the manuscript source against the pre-polish text (numbers, the 74 audited claims, cite
# keys, locked wording, placeholders, AI disclosure, banned phrases, em dashes) and of the compiled logs/PDFs.
manuscript-check:
	python3 paper/tools/check_bib.py
	python3 paper/tools/check_manuscript.py
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
# Release packaging previews (local only: nothing is published, tagged or archived)
release-preview:
	python3 scripts/make_release_preview.py
arxiv-preview:
	python3 scripts/make_arxiv_preview.py
