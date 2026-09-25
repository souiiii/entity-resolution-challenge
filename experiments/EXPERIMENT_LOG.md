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

### EXP-002 — Deterministic ranking, pruning and cap recovery (blocking-v2)

**Date:** 2026-09-25

**Goal / scope**
- Phase 3B blocking only: compare four deterministic rankings, prune per source,
  and recover capped evidence through specific conjunctions. No model or predictions.

**Data / split**
- Frozen fingerprint `1dd67d77fee74aa0b64477e10f09377ac22854f5eb13780f3aacc4baf94e3645`.
- Development only: 1,986,411 queries against the complete 5,034,616 S2 + 5,285,603 S3
  universe. Existing 40,170-query hash sample for five experiments; no validation labels.
- Final parameters frozen from the sample; exactly one full-development evaluation,
  computing every K simultaneously. Saved Phase 3A metrics reproduced exactly.

**Configuration**
- `experiments/configs/blocking_v2.json`: original nine channels unchanged;
  name token × address word added with frequency cap 10 and four rarest keys.
- Rank by exact address, exact name, summed inverse key frequency, channel families,
  smallest frequency, binary target ID. K=50 independently for S2 and S3.
- Rejected folded name × number (three sample links); cap 20 had diminishing returns.

**Measured full-development results**
- Blocking recall: 94.8223% (S2 94.8515%; S3 94.7950%).
- Candidates: 170,895,201, mean 86.032; 39.0064% reduction.
- Oracle blocking Macro F0.5 ceiling: 0.98059431; delta +0.00578173 versus Phase 3A.
- Recall change: +1.0254 percentage points.
- New recovery: 123,505 links before pruning;
  119,500 retained, with 48,984 old links lost.
- India 92.2261%; US 96.5577%.
- All-links entity recovery 85.6372%.
- Expanded union recall 95.5929%; pruning costs 52,989 links.

**Resources / checks**
- 187 tests passed. Full run 498.0 s; peak RSS 5,257 MiB.
- Stores and all retained indexes 5.321 GiB; free disk 23.94 GiB.
- No development pair TSV, test predictions, split/normalization changes or external data.
- Full curves, country/source breakdowns, sample comparisons and remaining failures:
  `BLOCKING_PHASE3B_REPORT.md`. Defer broad approximate retrieval; consider only a
  measured targeted pilot after downstream failure analysis.

**Country trade-off**
- India S2 loses 2,037 links; India all-links entity recovery loses 7,344 entities.
  The overall gain is concentrated in the US. No full-run retuning was performed.

### EXP-003 — Phase 4 pairwise features (features-v1)

**Date:** 2026-09-25

**Goal / scope**
- Deterministic pairwise features for frozen Phase 3B candidates; development-sample
  validation only. No matcher, thresholds, classifiers or predictions.

**Data / split**
- Frozen fingerprint `1dd67d77fee74aa0b64477e10f09377ac22854f5eb13780f3aacc4baf94e3645`.
- Existing 40,170-query development hash sample; development labels only; one
  complete sample analysis (plus smoke runs), then one rerun after removing five
  redundant/constant columns (other values unchanged).

**Configuration**
- `experiments/configs/features_v1.json` over `blocking_v2.json` (aggregate, K=50 per source).
- 42 float32 features: source, blocking evidence, 10 channel indicators, name (exact,
  RapidFuzz Indel/token-set, token overlap, length, script mismatch), address, integer
  address numbers, target-address missingness. NaN means undefined.

**Measured sample results**
- 3,454,436 pairs (131,709 positive, 3,322,727 negative) — identical to Phase 3B sample.
- Univariate AUC: aggregate rarity 0.975; address token-set 0.981; name token-set 0.840;
  number Jaccard 0.749. India names much weaker (char similarity 0.764 vs US 0.882) but
  numbers much stronger (number Jaccard 0.851 vs 0.694).

**Resources**
- Streaming 99.4 s (34,753 pairs/s, six workers); feature CPU 301.1 s; peak RSS 4,982 MiB
  main / 715 MiB worker; 95.6 MiB text offsets persisted; free disk 23.69 GiB.
- Projected full development (not run): ~1.4 h wall, ~4.1 CPU h; 26.7 GiB if materialized.
- 199 tests passed. See `FEATURE_REPORT.md`.
