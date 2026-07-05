<div align="center">

# Learned Cumulant Closure

**A physics-routed, permutation-equivariant learned third-cumulant closure for open, disordered
Tavis–Cummings superradiance — validated against exact references, with every hypothesis it *failed*
written down.**

![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)
![Julia](https://img.shields.io/badge/julia-≥1.9-9558B2?logo=julia&logoColor=white)
![QuTiP](https://img.shields.io/badge/exact_ref-QuTiP-1f77b4)
![status](https://img.shields.io/badge/science-frozen-2ca02c)
![license](https://img.shields.io/badge/license-MIT-blue)

</div>

---

## The one-paragraph version

Second-order cumulant expansions are the workhorse for simulating open collective light–matter systems too
large for an exact solve. This project studies where that workhorse breaks on a concrete, measurable observable
— the **superradiant burst peak** of an open, *disordered* cavity–emitter (Tavis–Cummings) ensemble — and builds
a small, structure-preserving **learned closure for the connected third cumulant** that recovers the exact-
reference burst in the tested regime. Everything is measured against exact QuTiP references where they exist;
nothing beyond that range is asserted as validated. Several tempting extensions were tested and **rejected**, and
they are kept in the record as evidence, not swept away.

## The headline, in one table

Signed burst-peak error vs. an exact master-equation solve, in the evaluated disordered open-cavity cells:

| closure | burst peak vs. exact | stability | cost |
|---|---|---|---|
| mean-field | no burst from these ICs | — | trivial |
| **2nd-order cumulant** | **overshoots ≈ +29%** | stable | cheap |
| κ4 = 0 (3rd-order) | undershoots ≈ −28% | **diverges** in the low-loss pocket | O(M⁴) |
| κ5 = 0 (4th-order) | overshoots then **diverges** | finite-time blow-up (M=3, both regimes) | large/erratic |
| **V7 (learned)** | **±1%** | stable | cheap, scales |

Among the tested zero-next-cumulant truncations through fourth order, none produced a stable, accurate burst
closure in these cells. V7 stays stable and matches the exact reference there. *(Scope: exact-referenced at the
tested molecule counts; behaviour beyond the exact ceiling is reported as estimate, not result.)*

## Why it fails, mechanistically

The burst-peak error is carried by **two connected third-cumulant channels** — photon-number–inversion
`⟨a†a σᶻ⟩` (dominant) and photon-polarization–inversion `⟨a σ⁺ σᶻ⟩` (secondary). Second order sets both to zero
(→ overshoot); κ4 = 0 mis-signs them (→ undershoot); V7 reconstructs them. See `FINDINGS.md` (Result 25) and
`results/mechanism_fig.png`.

## What's in here

```
exact_tc.py            exact open Tavis–Cummings (QuTiP mesolve / mcsolve references)
cumulant.py            2nd-order cumulant engine (validated == dense to ~1e-14)
scalable_engine.py     site-local, permutation-structured EOM for arbitrary M
closure_v3.py / _v6    the learned closures (V4 memoryless, V6 memory-augmented)
v7.py                  V7: parameter-routed dispatch between the two frozen closures
t3_truncation.py       κ4 = 0 third-order baseline
t4_truncation.py       κ5 = 0 fourth-order baseline (fairness test)
mechanism.py           connected-κ3 channel attribution + closed-loop ablation
FINDINGS.md            every numbered result (including the rejected hypotheses)
PROTOCOL.md            pre-registrations and their recorded outcomes
REPLICATION*.md        the cross-platform reproduction package (start with QUICKSTART)
```

## Reproduce the baseline in a separate codebase (Julia)

The reproduction package lets a collaborator re-run the **exact-vs-second-order burst-peak** comparison in
QuantumCumulants.jl + QuantumOptics.jl:

```bash
julia --project=. julia_setup.jl        # one-time install
julia --project=. replicate_cumulant.jl # runs the comparison, writes results/replication_cumulant.csv
```

Start with **[`REPLICATION_QUICKSTART.md`](REPLICATION_QUICKSTART.md)**; record results in
**[`REPLICATION_SIGNOFF.md`](REPLICATION_SIGNOFF.md)**; read the assumptions in
**[`REPLICATION_AUDIT.md`](REPLICATION_AUDIT.md)**. The Julia script has **not** been executed by the author and
is not asserted to be validated — that judgement is the collaborator's to record.

## Run the Python side

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python burst_pred.py       # exact vs 2nd-order vs V7 burst-peak comparison
python mechanism.py        # connected-κ3 channel attribution + ablation
```

## Scope and discipline (read this before judging)

- **Exact-referenced only where exact references exist.** Rollouts beyond that range are numerical reach, not
  demonstrated accuracy, and are labelled as such throughout.
- **Frozen after evaluation.** V7 and its two branches were frozen before final testing; pre-registrations and
  held-out disorder draws are in `PROTOCOL.md`.
- **Rejected hypotheses are kept.** A once-calibrated parameter-only routing criterion, an M-scaling story, and a
  disorder-dependence story were each tested and did not survive; they live in `FINDINGS.md` (Results 22–24, 26)
  as documented negatives.
- **No claim of exclusivity.** Tensor-network, trajectory, and higher-order-closure methods can address parts of
  this problem; this repository reports what these specific closures do in these specific cells.

## Provenance

The physics model, the
validation targets, and every reported number come from the code in this repository, which is runnable and
gated by the checks described in `PROTOCOL.md`.

## License

MIT — see [`LICENSE`](LICENSE).
