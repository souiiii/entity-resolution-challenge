# AGENTS.md

## Project Goal

Build a high-quality entity-resolution system for the supplied business-record matching challenge.

There are three sources:

- Source 1 is the reference/deduplicated source.
- Source 2 and Source 3 may contain records corresponding to Source 1 entities.
- A Source 1 entity may have zero, one, or multiple matches.
- There are no shared IDs across sources.
- Matching must be inferred only from supplied challenge data.

The objective is to maximize the challenge's Macro F0.5 score.

## Critical Evaluation Behavior

F0.5 weights precision more heavily than recall.

Therefore:

- False positive matches are especially costly.
- Never force a match just because one candidate is the closest.
- Correctly predicting no match for a singleton matters.
- Threshold selection must be tuned for Macro F0.5.
- Any claimed improvement must be verified on the fixed validation protocol.

## Data Restrictions

Do NOT use external data or lookup services.

Forbidden examples:

- Google Maps / Places
- OpenStreetMap lookups
- geocoding APIs
- web search for individual companies
- business registries
- external company/address databases

Only supplied challenge data may be used.

## Repository Rules

1. Keep raw challenge files unchanged under `data/raw/`.
2. Write derived files only to `data/interim/` or `data/processed/`.
3. Put reusable implementation in `src/`.
4. Put executable orchestration in `scripts/`.
5. Put experiment parameters in `experiments/configs/`.
6. Record meaningful experiments in `experiments/EXPERIMENT_LOG.md`.
7. Keep final artifacts in `output/`.
8. Add tests for deterministic behavior.
9. Do not silently change the validation split.
10. Prefer reproducible pipelines over ad-hoc notebook-only experiments.

## Intended Pipeline

1. Data inspection
2. Name normalization
3. Address normalization
4. Candidate generation / blocking
5. Pairwise feature generation
6. Strong deterministic baseline
7. Supervised model / ranker
8. Threshold optimization for Macro F0.5
9. Error analysis
10. Submission generation

## Blocking Requirements

Track at minimum:

- candidate-pair count
- average candidates per Source 1 entity
- median candidates
- p95 candidates
- maximum candidates
- blocking recall against train ground truth

Never optimize the matcher while ignoring blocking recall.

## Evaluation

Expose a single reproducible evaluation command:

```bash
python scripts/evaluate_model.py
```

Report:

- Macro F0.5
- precision
- recall
- false positives
- false negatives
- singleton-specific errors
- blocking recall
- threshold used

## Submission

The final pipeline must create:

- `output/candidate_pairs.tsv`
- `output/matching_results.tsv`

Integrate the official validator once available.

## Coding Expectations

- Python 3.11+
- type hints for public functions
- deterministic random seeds
- logging for pipeline code
- clear docstrings for non-trivial logic
- configuration-driven paths and thresholds where practical

## How Codex Should Work

For each task:

1. Read relevant existing modules.
2. Make the smallest coherent change.
3. Run relevant tests.
4. Run evaluation if matching behavior can change.
5. Record the experiment if results change.
6. Report measured effects, not assumptions.

Do not claim an approach is better unless validation confirms it.
