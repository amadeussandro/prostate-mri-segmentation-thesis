# Second-reviewer check of the literature audit: sample and analysis

Written 2026-11-12, before the sample was drawn. `scripts/literature_audit.py sample` records this
file's SHA-256 (LF line endings) in `sample.json`.

## Purpose

The audit (protocol.md) was screened and coded by one reviewer. A second reviewer, blind to the
first reviewer's decisions and codes, repeats the work on a random sample so that agreement can be
reported.

## Sample (drawn once, fixed seed)

- 8 of the 22 included studies (about a third), for eligibility and all 10 items.
- 10 of the 35 excluded records, for eligibility only (the reviewer codes the items too if they
  judge a record eligible).
- Drawn with Python `random.Random(20261112).sample` from each list of PMCIDs sorted as text. The
  18 records are then shuffled with the same generator, so that the reviewer cannot tell which
  were included.

## What the second reviewer receives

The protocol, the coding instructions, and the 18 PMCIDs with their PubMed Central links. The
instructions include the coding notes: general rules the first reviewer applied to borderline
wording, stated without reference to any study. The reviewer does not receive the first
reviewer's screening decisions, codes, quotations or flags, nor the manuscript text that reports
the result.

## Analysis

- Eligibility: percent agreement over the 18 records, and Cohen's kappa.
- Items, over the studies both reviewers judged eligible: percent agreement per item, and Cohen's
  kappa where at least one reviewer used more than one category for that item (kappa is undefined
  otherwise and is not reported).
- Agreement on the three key items (reassembly, inverse preprocessing, geometry check) is reported
  separately.

## Disagreements

Each disagreement is discussed by both reviewers against the article text and resolved. If a
resolution changes a first-reviewer code, the change is applied to `extraction.csv` or
`screening.csv` with a note, `summarize` is rerun, and the manuscript reports the reconciled
counts together with the agreement statistics before reconciliation.
