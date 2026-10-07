# History provenance: what version control can and cannot show

Written 2026-10-07. This file records how this repository's history came to be, what it can establish about
the order in which protocols and results were written, and what it cannot. No Git history has been
rewritten, rebased, re-dated or grafted to strengthen any claim. One earlier rewrite, made to remove
commit-message trailers, is disclosed below; nothing has been rewritten since, including during this audit.

## Two histories

**1. This repository** (`github.com/Dhruvjain35/learned-cumulant-closure`).
- The GitHub repository was created 2026-07-05 17:39 UTC.
- Its first 191 commits, `1cae32d` through `5fc4d8b`, were created by an import script on 2026-07-05
  between 12:35:30 and 12:35:42 CDT. The import replayed already-finished work file by file. Its commit
  timestamps record when the import ran, not when the work was done.
- Everything after the import (from 2026-07-05 14:05 CDT onward) was committed as the work happened. This
  covers Results 28–39 and every later edit.
- **One history rewrite has occurred, and it is disclosed here.** On 2026-07-05, before commit `d01687f`
  (17:45 CDT, "Re-map RESULTS_PROVENANCE commit hashes after history rewrite (trailer removal changed all
  commit hashes)"), the public history was rewritten to strip commit-message trailers. It was then
  force-pushed, and RESULTS_PROVENANCE was updated to the new hashes.
  - The rewrite changed every commit hash up to that point.
  - It preserved author and committer dates: the import commits still carry their 12:35 CDT times.
  - Its purpose was to change commit messages only, not to strengthen any scientific claim. Because the pre-rewrite history is not retained in this repository, that cannot now be independently checked.
  - Still, it means hashes cited anywhere before 2026-07-05 17:45 CDT do not resolve in this repository.
  - No other rewrite has occurred.

**2. The original development repository** (local only, `~/polariton-research`, path `spike/`).
- 69 commits, 2026-07-01 19:15 to 2026-07-09 19:34 CDT, with **no remote**.
- Its HEAD is `82f3aff54abe04c26233b772cf3f29f7d4da3ab4`. Because a Git commit hash commits to the
  complete history behind it, this hash fixes the content of every one of those 69 commits.
- This is the history that can corroborate protocol-before-result ordering for the work up to Result 22.
- It has not been published (see "Archival plan" below).

**Limit that applies to both.** Git timestamps are self-reported by the committing machine. They corroborate
the order in which things were committed. They are not tamper-proof proof of when text was written.
Stronger evidence would need an external timestamp (an archive DOI or a timestamping service), which
exists for nothing in this project yet.

## Pre-registration chronology

The "first commit" columns give the earliest commit containing the protocol text, and the earliest commit
containing the corresponding outcome. All times are CDT. "Private" is the original repository; "public" is
this one.

| Analysis (Result) | Protocol text first committed | Outcome first committed | Ordering shown by Git? | Classification |
|---|---|---|---|---|
| Results 1–9 (engines, learnability, transfer, retrain) | — (no pre-registration) | private `ea75384` 07-01 19:15 (single initial commit) | n/a | exploratory development |
| V3 pilot (R10) | private `1d15e84` 07-01 19:25 | private `fa5e039` 07-01 20:05 | yes | pre-registered, timestamp-corroborated (private) |
| Regime campaign (R11) | private `1d15e84` 07-01 19:25 | private `fc885e1` 07-01 21:22 | yes (the claim-class amendment came after and is labeled so) | pre-registered, timestamp-corroborated (private) |
| Support-risk indicator (R13; restated as R33) | private `c3247dd` 07-01 22:03 | private `3a9b097` 07-01 22:27 / `b370c22` 22:41 | yes | pre-registered, timestamp-corroborated (private) |
| 2x2 decision tree (R14) | private `fa9366b` 07-01 22:45 | private `7f81060` 07-01 23:05 | rule precedes the verdict; runs launched earlier (21:36) | decision rule locked before the verdict, not before the runs |
| V5 control (R16–R17) | private `5a73cee` 07-01 23:50 | private `249b5d8` 07-02 00:39, `e1ee0b4` 04:36 | yes | pre-registered, timestamp-corroborated (private) |
| V6 memory closure (R17–R18) | private `651f81e` 07-02 00:35 | private `e1ee0b4` 07-02 04:36, `b876516` 06:07 | yes | pre-registered, timestamp-corroborated (private) |
| V7 router (R19) | private `9fd9a9e` 07-02 14:09 | private `2ee043b` 14:35 (seam), `380aab5` 15:23 (final) | yes | pre-registered, timestamp-corroborated (private) |
| Baseline ladder (R20) | private `2ce0537` 07-02 16:23; final lock `8e7c8db` 16:48 | private `aea45cf` 19:40 (dev), `cc84a71` 21:39 (final) | yes | pre-registered, timestamp-corroborated (private) |
| Interpretability (R21) | private `6be882e` 07-03 01:14; rule `b598498` 01:21 | private `02588ed` 07-03 03:10 | yes | pre-registered, timestamp-corroborated (private) |
| Memory criterion, Step 0 | private `5a3cef3` 07-04 14:07 | same commit `5a3cef3` | no | analysis of existing data; not prospectively verifiable |
| Memory criterion, Step 1 (R22) | private `5a3cef3` 07-04 14:07; wording lock `460d324` 14:27 | private `2f48029` 14:19 (rebuilt run), `7c03e6d` 16:17 (verdict) | yes | pre-registered, timestamp-corroborated (private) |
| **Burst vs M (R24)** | private `315912f` 07-09 19:34:47; public import 07-05 12:35 | private `674f3e8` 07-09 19:34:46; public import 07-05 12:35 | **no** | **protocol documented, prospective timing not verifiable; treated as non-preregistered** |
| **Mechanism (R25)** | private `315912f` 07-09 19:34:47; public `e692d11` 07-05 12:35:35 | private `f7c0097` 07-09 19:34:46; public `f3af19b` 07-05 12:35:34 | **no** | **as R24; also deviated from its written protocol (below)** |
| **Disorder sweep (R26)** | private `315912f` 07-09 19:34:47 | private `50863a3` 07-09 19:34:47 | **no** | **as R24** |
| **Fourth-order fairness test (R27)** | private `315912f` 07-09 19:34:47 | private `30f5524` 07-09 19:34:47 | **no** | **as R24** |
| Post-freeze audits (R28–R32) | — | public 2026-08-09 | n/a | exploratory audits of the frozen model |
| Zero-disorder reference, Step 0 (R34) | public `cdca2e3` 2026-09-15 18:24 | public `05e378b` 09-15 19:23 | yes | pre-registered, timestamp-corroborated (public) |
| σ-sensitivity, M = 16 ladder, disordered M = 12, disordered ladder (R35–R38) | — (no separate pre-registration) | public 09-15 to 09-17 | n/a | exploratory follow-ups |
| Common-carrier audit (R39) | public `78e9a38` 2026-10-05 00:14 | public `ccc7414` 2026-10-05 | yes | pre-registered, timestamp-corroborated (public); its Part A companion is post-hoc and labeled so |
| Positivity-architecture option ("V8 Step 1") | public `45690c0` 2026-09-18 | — | n/a | registered option, **not started**; hard stop 2026-12-31 |

### Results 24–27 in detail

- **The public import (07-05 12:35) already contains these four analyses.** Scripts, logs and the
  PROTOCOL sections that describe them as "written BEFORE any run" all appear within the same few seconds
  of the import.
- **The private repository** has no commits between 2026-07-04 16:17 (Result 22) and 2026-07-09 19:34:46.
  At 19:34:46–47 a batch of six commits records the same work, in this order:
  1. the Result 24–27 code and logs;
  2. `315912f` "FINDINGS + PROTOCOL: write up Results 23-27 and lock the frozen-modeling state";
  3. the replication package.
- That batch is a **backfill** of work that already existed (it was already in the public import four days
  earlier). Its internal commit order reflects how the backfill was committed, not the order in which the
  protocol text and the computations were written.
- **Conclusion:** no surviving record establishes that the Result 24–27 protocols were written before their
  computations. The author's account is that they were specified in-session before each run, and the
  repository has always said so. Without a record that can verify it, those four results are classified for
  publication as **non-preregistered analyses that followed a documented decision rule**. Their scientific
  content is unchanged. Only the strength of the prospective-protocol evidence is restated.
- **Result 25 additionally deviated from its own written protocol** in three respects: the adjudicating
  metric, a decision threshold that appears only in code, and a confirmation run at M = 5 that was never
  done. These are recorded in `docs/FINDINGS.md` (correction note under Result 25) and in
  `docs/PUBLICATION_CLAIM_AUDIT.md` (P2).

## Archival plan (not executed)

- **Safety review of the original repository (2026-10-07):**
  - No credential patterns appear in any revision (API-key, token, private-key and password searches all
    return zero hits).
  - Every commit is authored by the project author, with a local-machine address.
  - **It requires sanitization before any public release.** Alongside `spike/` (this project) it tracks
    material from a separate collaboration that is not this project's to publish:
    - planning documents (`01-…`–`04-…md`) and an Overleaf draft (`overleaf-pilot-v2.tex`);
    - literature notes (`papers/`), one of which contains a third-party email address;
    - a small script and folder (`check_nmax.py`, `sim/`);
    - a gitlink to a collaborator's repository (`Sharingan-Cavity`).
  - It also holds large training arrays (single files up to ~82 MB).
- **Recommended form:**
  - Deposit an immutable, clearly labeled "historical development repository, not canonical current
    source": a `git bundle` with its SHA-256, for example on Zenodo.
  - Restrict it to the `spike/` history, with every author and committer date preserved, and publish a map
    from original to filtered commit hashes.
  - State that the filtered copy is derived from the commit `82f3aff…` recorded above. If a referee asks,
    the unfiltered history can be checked against that hash privately.
  - Large training arrays may be excluded and listed by hash.
- **Do not merge or graft it into this repository's `main`.** Doing so would present a reconstructed history
  as if it had always been canonical. That is exactly the kind of after-the-fact ordering this note exists
  to avoid. The two histories stay separate and are linked by this document.
