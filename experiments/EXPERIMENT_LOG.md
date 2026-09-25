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

### EXP-004 — Phase 5 deterministic matcher baseline and first submission (matcher-v1)

**Date:** 2026-09-25

**Goal / scope**
- First complete pipeline: Phase 3B candidates → Phase 4 features → explicit prefilter →
  deterministic matcher → official files. No learned model.

**Data / split**
- Frozen fingerprint `1dd67d77fee74aa0b64477e10f09377ac22854f5eb13780f3aacc4baf94e3645`.
- Tuning on the 40,170-query development hash sample only (≈495 small-grid configurations in
  six documented steps). Frozen config then run once on validation (220,410 queries against
  the complete training S2 + S3); no retuning.

**Configuration** (`experiments/configs/matcher_v1.json`)
- Prefilter: (no number conflict and (address ≥ 0.60 or name ≥ 0.75)) or blocking rank ≤ 5.
- Rules: numbers agree & name ≥ 0.70 & address ≥ 0.60; numbers agree & script mismatch &
  address ≥ 0.85; numbers unavailable & name ≥ 0.85 & address ≥ 0.65; then keep matches
  with rarity ≥ 0.10 × the query's best acceptable rarity.

**Results**
- Development sample: Macro F0.5 0.85636; 24.73 candidates/S1 (Phase 3B 86.00); candidate
  recall 94.02%; oracle 0.97737.
- Validation (single frozen run): Macro F0.5 **0.85376**; TP 567,270, FP 35,783, FN 194,322;
  singletons 9,483/12,360 correct; 24.78 candidates/S1; recall 94.01%. US 0.88947, India 0.80001.

**First test submission — matcher-v1** (not uploaded)
- `output/matching_results.tsv`: 1,732,544 rows, 5,030,493 matches (112,113 empty), 87,280,825 bytes,
  sha256 `4cb9a2327cd6cecb6a131dba5b7756a2c4d8084cfa212dc7a6efe383fc1cdaee`.
- `output/candidate_pairs.tsv`: 44,419,444 candidates (mean 25.64, p50 22, p95 54, max 100),
  594,841,148 bytes, sha256 `bd0cd868dfaa017e047637994ef38fa107d4c28e50e63f575d7512fabcb77394`.
- Official validator: matching file PASS with `--check-ids`; both files PASS via
  `scripts/validate_outputs.py` (official per-shard checks + global invariants).
- Known risk: French town-name token inflation (max 85 matches for one entity; 2,510 entities
  with >10 matches). Not tuned on unlabeled test data.

**Resources**
- Test run 4,990 s; validation 593 s; free disk 17.83 GiB after ~5.9 GiB of test artifacts/outputs.
- 210 tests passed. See `BASELINE_REPORT.md`.

### EXP-005 — Phase 6 bounded supervised matcher (supervised-v1)

**Date:** 2026-09-25

**Scope / protocol**
- Existing Phase 3B candidates, exactly 42 Phase 4 features, and unchanged Phase 5 prefilter.
- One query hash split of the 40,170-query development sample: salt `supervised-query-split-v1`, tune fraction 0.20.
- Train 32,176 queries / 794,459 screened pairs (104,450 positive); tune 7,994 queries / 198,913 pairs (26,105 positive). All screened negatives retained.
- Exactly two LightGBM 4.7.0 CPU configurations: 31/63 leaves, learning rate .05, 500 rounds, minimum leaf rows 100, column fraction .9, L2 2, six threads, seed 42. Tune-only logloss early stopping, patience 50; both reached 500 rounds.
- One global coarse-to-fine probability threshold strategy (.05/.005); canonical Macro F0.5 agrees exactly with the fast sweep.

**Development selection**
- Same-query deterministic baseline: **0.8599705764462935**.
- 31 leaves: **0.9235863645259758**, threshold .705.
- Selected 63 leaves: **0.9259311292573912**, threshold **.660**; gain **+0.0659605528110977**, exceeding the +.01 gate.
- Precision .947707 → .972000; singleton false merges 93 → 56; India Macro .805125 → .892899. US singleton false merges 19 → 26, a modest recorded regression.
- Candidate regression exact: 993,372 screened pairs, mean 24.7292, recall .94023896, oracle .97737263.

**One frozen validation confirmation**
- Frozen model saved without refitting; no validation-based tuning; no repeat validation.
- Macro F0.5 **0.9253015854145195**, versus Phase 5 **0.853761517044282**: **+0.07154006837023741**.
- TP 659,664; FP 19,524; FN 101,928; precision .97125391; recall .86616456.
- Correct singletons 10,953/12,360; false merges 1,407 (Phase 5: 2,877).
- India .89697321; US .94412299. US singleton false merges 654 → 699; no retuning.
- Identical candidate stage: 5,462,251 pairs, mean 24.78223, median 21, p95 51, max 100, recall .94014118, oracle .97747447.

**Artifacts / resources**
- Model: `output/supervised_v1/model.txt`, 3,519,918 bytes; SHA256 `744aa05566e3d77f0c10c9d01a450a8bf518fd3142b97efe87629b41d52b9e75`.
- Development 469.29 s / 4,277.10 MiB peak main RSS; validation 875.89 s / 4,558.42 MiB peak main RSS. Worker aggregate peak unavailable under forkserver; minimum observed system available memory 4,384.26 MiB development / 5,317.64 MiB validation.
- Minimum free disk 17.55698 GiB. No feature matrix persisted; validation features streamed.
- Both Phase 5 submission hashes unchanged. No external data, country features/rules, test inference, leaderboard submission, or Phase 7 work.
- Detailed results, feature importance, limitations, and reproduction: `SUPERVISED_REPORT.md`.
- Final full suite: 218 passed in 20.14 s; Phase 6 complete, with no post-validation optimization.
