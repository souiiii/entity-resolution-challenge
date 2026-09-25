# Experiment Log

## Template

### EXP-000 — Name

**Date:** YYYY-MM-DD

**Goal**
- What is being tested?

**Data / Split**
- Fixed split details.

**Blocking**
- Methods:
- Candidate count:
- Blocking recall:

**Features**
- Features used.

**Model**
- Algorithm:
- Parameters:

**Threshold**
- Value:

**Results**
- Macro F0.5:
- Precision:
- Recall:
- False positives:
- False negatives:
- Singleton errors:

**Notes**
- Main failure modes.
- Next experiment.

### EXP-001 — Deterministic blocking baseline (blocking-v1)

**Date:** 2026-09-25

**Goal**
- Measure blocking recall and candidate volume of deterministic key channels (Phase 3A).

**Data / Split**
- Frozen split `1dd67d77fee74aa0b64477e10f09377ac22854f5eb13780f3aacc4baf94e3645`; development
  Source 1 only (1,986,411 queries; 3,325,308 S2 and 3,551,465 S3 links) against the complete
  training S2 + S3 corpus. Validation labels not used. One sample configuration, one full run.

**Blocking**
- Methods: `experiments/configs/blocking_v1.json` — exact address/name/folded/folded-compact, rare
  name token, name-token pairs, name token × address number, address number × word, rare address word;
  all same-country.
- Candidate count: 280,185,604 pairs; mean 141.1, p50 129, p95 298, p99 388, max 885 per S1.
- Blocking recall: 93.80% overall (S2 93.81%, S3 93.79%; India 92.09%, US 94.94%).

**Features / Model / Threshold**
- None (candidate generation only).

**Results**
- Entities with all links recovered 83.80%; with none 1.01%. See `BLOCKING_REPORT.md`.

**Notes**
- 99.6% of misses share a key only above frequency caps; 1,674 share no key.
- Next: candidate pruning/ranking and more specific conjunctive keys (Phase 3B).
