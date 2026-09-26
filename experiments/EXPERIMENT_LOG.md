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

### EXP-006 — Phase 7 bounded decision optimization (decision-v1)

**Date:** 2026-09-26

**Scope / protocol**
- Frozen Phase 6 LightGBM model, 42 features, Phase 3B candidates and Phase 5 screen; no retraining or upstream changes.
- Only 7,994 held-out Phase 6 tune queries. Exact baseline reproduction: Macro F0.5 **0.9259311292573912**, 198,913 screened pairs, original membership/schema/model hashes.
- One query hash split: `decision-v1-confirm-split`, confirm fraction .4; search 4,788 queries / 119,500 pairs; confirm 3,206 queries / 79,413 pairs.
- Exactly **121** fixed search configurations: global 26, relative-to-best 36, margin 30, singleton abstention 29. One search winner per family evaluated on confirm; no adaptive refinement or source/country rules.

**Result / stopping decision**
- Best robust candidate: probability >= .660 and >= .70 × query best across both sources.
- Search gain +.0003060051277050668; confirm gain **+.00018917308338517458**.
- Complete tune **0.9261902788016474**, gain **+.00025914954425620973**, below the required +.002.
- TP 24,126 → 24,000; FP 695 → 630; FN 3,622 → 3,748; precision .97199952 → .97442144; recall .86946807 → .86492720.
- Correct singletons 410 and false merges 56 unchanged; exact matches 5,000 → 4,962.
- Best global threshold stayed .660. Margin and abstention winners lost on search and confirm; rejected without further variants.
- **Phase 7 gate failed. Retain Phase 6 global .660. No Phase 7 frozen validation was run or started.** No test inference/submission, leaderboard tuning, or Phase 8 work.

**Resources / artifacts**
- 167.42 s; peak main RSS 4,004.31 MiB; minimum observed system available memory 6,057.97 MiB; minimum free disk 16.94548 GiB. Worker/aggregate peak unavailable under forkserver.
- Streamed features; compact probabilities only in RAM; no feature or score cache persisted.
- Candidate policy/config SHA256: `3360d6dd30a0bb6f50ed377a37995d8a38e4d68a800db201ffbf57c9ef884680`; candidate is not adopted.
- Existing model, metadata and both submission hashes unchanged. Detailed diagnostics and limitations: `DECISION_REPORT.md`.
- Phase 7 does not change the ~.9775 validation candidate oracle; candidate recall is deferred to Phase 8.
- Final full suite: **237 passed** in 28.77 s. No post-gate search or validation.

### EXP-007 — Phase 8 candidate recall recovery (blocking-v3)

**Date:** 2026-09-26

**Goal / scope**
- Candidate generation only: raise final-candidate recall without exploding candidate counts.
  No retraining, threshold work, test inference or submission.

**Data / split**
- Frozen fingerprint `1dd67d77fee74aa0b64477e10f09377ac22854f5eb13780f3aacc4baf94e3645`.
- Design on the 40,170-query development sample (development labels only); one full-development
  run (1,986,411 queries); one candidate-only validation run (220,410 queries).

**Loss accounting (sample, Phase 5 final set, 8,298 missed of 138,853)**
- A no shared key 32 (0.4%); B truncated buckets 2,736 (33.0%); C frequency cap 2,936 (35.4%);
  C key limit 417 (5.0%); D rank > 50 1,023 (12.3%); E Phase 5 prefilter 1,154 (13.9%; 92% number conflicts).

**Designs (three, no fourth)**
- D1 union re-ranked by cheap hashed score top-M: worse than Phase 5 at equal size.
- D2 deep retrieval (caps lifted to index keep) + cheap score top-M: 95.29% at 50/query.
- D3 (selected): Phase 5 screen + conflict rescue (N >= .8, A >= .6) + rank rescue (score >= 1.5
  or best) + 8 best deep-only per query and source. Config `experiments/configs/blocking_v3.json`
  (sha256 42ab3b48abfacb758d77ed9172d68791832a4cb9ed649da76fe00250c8c9ca65).
- Rejected inside D3: IDF-weighted overlap ranking (slightly worse); deep channel restriction.

**Results**
- Sample: recall 94.024% -> 96.108%; oracle 0.97737 -> 0.98598; mean 24.73 -> 42.44.
- Full development: 93.988% -> 96.054% (+2.07 pp); oracle 0.97749 -> 0.98592; mean 24.79 -> 42.49
  (p99 87, max 185); +142,097 links for +35.2M pairs; India 93.84%, US 97.53%.
- Validation (single run): 94.014% -> **96.101%** (+2.09 pp); oracle 0.97747 -> **0.98606**;
  mean 24.78 -> **42.48** (p99 86, max 178); S2 96.07%, S3 96.13%; India 93.94%, US 97.54%.
- Criteria: recall +2 pp and mean <= 50 met; oracle >= 0.990 not met (deep pool of 888/query reaches 0.9932).
- Residual (sample, 5,404 missed): B 50.6%, C 35.0%, D 8.0%, E 5.9%, A 0.6%.

**Resources**
- Full development 8,168 s; validation 905 s; peak RSS 6,480 MiB (with mapped files);
  minimum free disk 15.21 GiB; no new index; temporary 1.1 GB sample pool deleted.
- See `CANDIDATE_RECALL_REPORT.md`. Phase 9 must retrain on the blocking-v3 final set.

### EXP-008 — Phase 9 large-bucket recovery and candidate ranker (blocking-v4)

**Date:** 2026-09-26

**Goal / scope**
- Candidate generation only: recover index-truncated (Category B) links with a supplemental overflow index and
  a separate candidate ranker. No final-matcher retraining, threshold work, test inference or submission.

**Data / split**
- Frozen fingerprint `1dd67d77fee74aa0b64477e10f09377ac22854f5eb13780f3aacc4baf94e3645`.
- Design on the 40,170-query development sample; ranker query split salt `phase9-candidate-ranker-v1`, 0.7:
  28,035 training / 12,135 held-out queries.

**Blocking**
- Overflow `overflow-csr-v1` for buckets <= 1000: 395,057,544 bytes (376.76 MiB) vs 395,042,844 estimated.
- Broad retrieval (deep + overflow), sample: 282,598,169 pairs (7,035/query); recall 0.990364; oracle 0.996872;
  +4,066 links over Phase 8.
- Ranker: LightGBM (180 trees, 31 leaves, seed 901), 34 features, 2,792,408 rows (95,873 positives); 37.98 s fit;
  model `output/candidate_ranker_v1.txt` sha256 13d4d8415124e67539fcd595ce1e3e74de8cf588b2ad1b2eddc03638b777fc1e.
- blocking-v4 = Phase 8 set + 8 best ranker rescues per query and source; frozen
  `experiments/configs/blocking_v4.json` (sha256 e4916142ccf3aaa96fc1266068e03d5c66d8e4cf0aefabaf9fe07b900b681347).

**Results**
- Held-out +8: recall 0.96114 -> **0.98644**; oracle 0.98584 -> **0.99538**; mean 42.49 -> **58.49**
  (median 55, p95 86, p99 103, max 139); +1,064 links for +194,158 pairs.
- Sample +8: 0.98656 / 0.99551 / 58.44 (+3,538 links, +642,691 pairs). +5 misses both gates; +10 exceeds 60/query.
- Production iterator reproduces the stored sample +8 selection pair-for-pair on an unaligned 2,000-query slice.
- Full-development confirmation: started, then terminated before any shard completed after the deadline change
  (estimated 17-22 h with 4 workers). Frozen validation: not run (estimated >= 1.9 h, above the 60-minute budget).
- blocking-v4 is therefore unconfirmed on full development and validation.

**Resources**
- Sample evaluation 4,415 s, peak RSS 5,265 MiB (with mapped files); minimum free disk 14.52 GiB.
- Throughput ~6.1 s per 100 query-sources per process; test blocking-v4 generation would take ~15-20 h.
- Recommendation: use validated blocking-v3 for the deadline matcher/submission. See `LARGE_BUCKET_RECOVERY_REPORT.md`.
