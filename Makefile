# Learned Cumulant Closure — convenience targets. Always run from the repository root.
# Python entry points go through run.py (a flat-import path shim so the reorganized layout keeps working);
# the Julia reproduction is standalone.
PY ?= python

.PHONY: help test gates burst mechanism fourth-order replicate

help:
	@echo "make test          # import smoke test (fast, no simulation)"
	@echo "make gates         # model-convention and EOM-identity gates (the derivation checks)"
	@echo "make burst         # exact vs 2nd-order vs V7 superradiant burst-peak comparison"
	@echo "make mechanism     # connected-kappa3 channel attribution + closed-loop ablation"
	@echo "make fourth-order  # exact vs 2nd vs 3rd vs 4th vs V7 fairness test"
	@echo "make replicate     # Julia cross-platform reproduction (needs Julia; see docs/REPLICATION_QUICKSTART.md)"
	@echo ""
	@echo "run any module:    $(PY) run.py src/<subpackage>/<module>.py"

test:
	$(PY) run.py tests/test_imports.py
# Deterministic pass/fail gates only. validate_model.py is a separate physics probe: its gates 1-2
# (vacuum Rabi, collective sqrt(M)) are analytic pass/fail, but its gate 3 reports a premise
# diagnostic rather than a pass/fail, so it is run on its own rather than folded in here.
gates:
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
