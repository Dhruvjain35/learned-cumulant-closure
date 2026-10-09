# Learned Cumulant Closure — convenience targets. Always run from the repository root.
# Python entry points go through run.py (a flat-import path shim so the reorganized layout keeps working);
# the Julia reproduction is standalone.
PY ?= python

.PHONY: help test gates burst mechanism fourth-order replicate portable-export portable-test figures paper-generate paper supplement

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
	@echo "make paper-generate # regenerate paper/generated/{abstract,body}.tex from docs/MANUSCRIPT_DRAFT.md"
	@echo ""
	@echo "run any module:    $(PY) run.py src/<subpackage>/<module>.py"

test:
	$(PY) run.py tests/test_imports.py
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
paper-generate:
	python3 paper/tools/md_to_revtex.py
paper:
	python3 paper/tools/check_bib.py
	cd paper && $(TECTONIC) -X compile --keep-logs main.tex
supplement:
	$(PY) run.py paper/tools/build_supplement_tables.py
	cd paper && $(TECTONIC) -X compile --keep-logs supplement.tex
