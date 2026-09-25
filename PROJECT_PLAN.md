# Implementation Plan

## Phase 1 — Data Understanding
Identify all official files, IDs, name/address columns, nulls, duplicates, ground truth, sample submission, and validator. Produce `DATA_REPORT.md`.

## Phase 2 — Normalization
Implement conservative normalization for names and addresses while preserving raw fields.

## Phase 3 — Blocking
Build several high-recall candidate-generation methods and union them. Measure blocking recall.

## Phase 4 — Pairwise Features
Add fuzzy name/address similarity, token overlap, character similarity, numeric-address agreement, locality/postcode features where available, and missingness flags.

## Phase 5 — Deterministic Baseline
Build a strong weighted/rule baseline and tune thresholds for F0.5.

## Phase 6 — Supervised Model
Try compact tabular models first. Compare every model against the deterministic baseline.

## Phase 7 — Threshold Optimization
Tune specifically for the official Macro F0.5 metric and inspect false-positive-heavy regimes.

## Phase 8 — Error Analysis
Bucket failures into blocking failures, aliases, typos, shared-address collisions, missing data, transliteration, and threshold failures.

## Phase 9 — Advanced Improvements
Only after the classical system is strong: ensembles, hard-negative mining, learned representations, or source-specific calibration.

## Phase 10 — Final Submission
Generate `candidate_pairs.tsv` and `matching_results.tsv`, run tests, evaluation, and the official validator.
