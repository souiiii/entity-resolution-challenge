# Phase 9 — Large-bucket recovery and candidate ranker (blocking-v4)

Candidate generation only. Branch `phase9-large-bucket-recovery`; starting commit
`5aa27474a9a18f4daa8324f0977591b8fcf2c816` (Phase 8, blocking-v3). All design labels come from the existing
40,170-query development sample. The frozen query split and all Phase 8 configurations/artifacts are preserved.
No final matcher training, threshold optimization, test inference, France inspection, enrichment or submission is
part of this phase.

**Outcome.** The +8 rescue policy (blocking-v4) is frozen from sample and held-out evidence: held-out recall
**0.98644**, oracle Macro F0.5 **0.99538**, mean **58.49** candidates/query, against Phase 8's 0.96114 / 0.98584 /
42.49 on the same queries. Because of the competition deadline, the full-development confirmation was started and
then **terminated before any shard completed**, and the frozen validation run was **not run**. It was estimated at
≥ 1.9 h, above the 60-minute budget. blocking-v4 therefore has no full-development or validation confirmation.
Section 24 recommends Phase 8 blocking-v3 for the deadline submission.

## 1. Phase 8 starting point

| Partition | Recall | Oracle Macro F0.5 | Mean candidates | Median | p90 | p95 | p99 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Development sample (40,170) | 0.961081 | 0.985977 | 42.438 | 39 | 62 | 70 | 87 | 123 |
| Full development (1,986,411) | 0.960544 | 0.985925 | 42.495 | 39 | 62 | 70 | 87 | 185 |
| Frozen validation (220,410) | 0.961011 | 0.986063 | 42.480 | 39 | 62 | 70 | 86 | 178 |

Full development has 84,412,160 Phase 8 pairs and 6,605,445 of 6,876,773 links found. Validation has 9,363,000
pairs and 731,898 of 761,592 links found.

## 2. Residual-loss analysis

The sample has 138,853 true links. Phase 8 recovers 133,449 and misses 5,404:

| Category | Missed links | Share |
| --- | ---: | ---: |
| A — no shared deterministic key | 32 | 0.6% |
| B — bucket truncated by index storage | 2,736 | 50.6% |
| C — query-time frequency cap | 1,792 | 33.2% |
| C — rare-key limit | 96 | 1.8% |
| D — union ranking | 430 | 8.0% |
| E — Phase 5 prefilter | 318 | 5.9% |

Category B dominates. `split_oversized` in `PostingIndex` removes **all postings** of a bucket above `index_keep`,
keeping only `large_keys`/`large_counts`. It does not keep the first 200 records. Nine channels keep buckets through 200;
`name_x_address_word` keeps only through 50. Query-time frequency caps and rare-key limits are separate restrictions.
`deepen` removes key limits and lifts caps to the storage limit. Phase 8 keeps its Phase 5 screen and adds conflict,
union-rank and eight deep rescues per source.

## 3. Category-B recovery by bucket range

Each link is assigned once to its smallest useful shared bucket; channel ties follow configuration order. The
51–200 row exists because one base index stops at 50. Name similarities are folded token-set similarities across
all scripts; address similarities are canonical token-set similarities.

| Best bucket | Unique links | Script mismatch | Median name | Median address | Number missing / agree / partial / conflict |
| --- | ---: | ---: | ---: | ---: | --- |
| 51-200 | 65 | 8 | 0.871 | 0.851 | 32 / 17 / 5 / 11 |
| 201-300 | 371 | 157 | 0.432 | 0.894 | 162 / 141 / 13 / 55 |
| 301-500 | 439 | 216 | 0.176 | 0.904 | 190 / 175 / 19 / 55 |
| 501-750 | 304 | 135 | 0.222 | 0.889 | 147 / 96 / 10 / 51 |
| 751-1000 | 251 | 95 | 0.435 | 0.902 | 133 / 84 / 4 / 30 |
| >1000 | 1,306 | 494 | 0.462 | 0.863 | 729 / 260 / 31 / 286 |

## 4. Cumulative theoretical recovery (overlap-corrected, pre-build)

| Maximum frequency | Additional unique links | Share of Phase 8 misses | Required postings | Array bytes | Projected recall | Projected oracle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 300 | 436 | 8.07% | 44,319,312 | 183,062,064 | 0.983205 | 0.994302 |
| 500 | 875 | 16.19% | 66,300,282 | 271,900,952 | 0.986367 | 0.995419 |
| 750 | 1,179 | 21.82% | 83,094,850 | 339,520,792 | 0.988556 | 0.996150 |
| **1000 (selected)** | **1,430** | **26.46%** | **96,911,515** | **395,042,844** | **0.990364** | **0.996872** |

The 1,000 cap was chosen before ranker training. The 500 cap leaves only 0.000419 oracle margin before compression;
the 1,000 cap gives 0.001872 at a small storage cost. All eligible overflow keys are kept; none are selected
from positive labels. Per-source/channel estimates for all four thresholds are in
`data/interim/candidates/phase9_large_bucket_analysis.json`. At the selected cap:

| Source / channel | Attributed links | Independent links | Postings | Array MiB |
| --- | ---: | ---: | ---: | ---: |
| S2/address_number_x_word | 294 | 295 | 6,665,504 | 25.68 |
| S2/address_rare_word | 339 | 344 | 3,159,640 | 12.16 |
| S2/name_rare_token | 18 | 24 | 2,000,339 | 7.69 |
| S2/name_token_pair | 70 | 73 | 3,688,189 | 14.21 |
| S2/name_token_x_address_number | 3 | 6 | 2,437,934 | 9.40 |
| S2/name_x_address_word | 30 | 31 | 30,052,081 | 117.47 |
| S3/address_number_x_word | 218 | 226 | 6,882,261 | 26.52 |
| S3/address_rare_word | 276 | 281 | 3,023,197 | 11.64 |
| S3/name_exact | 1 | 1 | 8,604 | 0.03 |
| S3/name_rare_token | 24 | 29 | 1,842,804 | 7.09 |
| S3/name_token_pair | 80 | 94 | 3,577,451 | 13.79 |
| S3/name_token_x_address_number | 12 | 44 | 2,662,257 | 10.26 |
| S3/name_x_address_word | 65 | 73 | 30,862,340 | 120.62 |

The exact-name/exact-address channels hold < 0.1 MiB of overflow each, and every link they recover is already
attributed to another channel.

## 5. Overflow-index architecture

Phase 9 stores only the missing oversized buckets (≤ 1,000) as a supplemental `overflow-csr-v1` index per
source and channel. Each index has a unique uint64 key dictionary, int64 CSR offsets and int32 target rows. The base
indexes are unchanged and no base posting is duplicated. Target records stream in bounded 10,000-row batches, and known
bucket counts allocate exact slots, so no external-sort staging is needed. Files are mmap-backed, source-separated,
tied to the source input hash, store version, base index ID and maximum, and reused when the identity matches.
Incomplete builds are removed on failure; unexpected staging directories and stale indexes are rejected.
`BroadRetriever` unions the existing deep pool with overflow hits (caps lifted to 1,000, no key limit), deduplicates
pairs, and keeps the channel mask, minimum shared frequency, summed rarity and key-hit count.

## 6. Estimated vs actual storage

| | Bytes | MiB |
| --- | ---: | ---: |
| Estimated arrays | 395,042,844 | 376.74 |
| Actual files incl. headers/metadata | 395,057,544 | 376.76 |

Initial construction took 157.82 s at 4,008 MiB peak RSS. Every build and retrieval batch checks the 10.5 GiB
free-disk reserve, and heavy runners disable core dumps. The normalized text cache (`candidate-text-v1`, all three
training sources, 929.3 MB of text plus offsets) is the only other new large artifact.

## 7–8. Broad retrieval ceiling and pair count (sample)

| Retrieval universe | Pairs | Mean/query | Median | p95 | Max | Recall | Oracle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 8 final set | 1,704,717 | 42.44 | 39 | 70 | 123 | 0.961081 | 0.985977 |
| deep (existing indexes) | 35,667,649 | 887.92 | 765 | 1,951 | 10,579 | 0.980065 | 0.993206 |
| **broad (deep + overflow ≤ 1000)** | **282,598,169** | **7,035.06** | 6,004 | 15,719 | 59,320 | **0.990364** | **0.996872** |

The deep pool exactly reproduces Phase 8's 35,667,649 pairs and 136,085 recovered links. Overflow adds 1,430 links. The
broad universe contains 137,515 true links, **4,066 more than Phase 8**. It leaves 1,306 Category-B links above 1,000
and 32 links with no shared key. Broad recall is 0.99022 for S2 and 0.99050 for S3, and 0.98700 for India and 0.99261 for US.

The first ceiling pass overwrote per-source candidate counts; the cached-index pass (128.6 s, 2,983 MiB) corrected
this and is reported above. The broad universe is a temporary retrieval pool and is **never** an official candidate
set.

## 9. Candidate-ranker features

The 34 float32 features (`candidate-ranker-features-v1`) are:

- Source 3 indicator.
- Shared name tokens, name Jaccard and name containment; shared address words, address-word Jaccard and containment.
- Number agreement state (missing/agree/partial/conflict); exact folded/compact name and canonical address agreement.
- Minimum shared frequency, summed inverse frequency, key-hit count, channel count and distinct family count.
- Overflow-only provenance and minimum-frequency band.
- Phase 8 cheap score, and query-relative cheap-score and rarity ranks.
- Folded-name character and token-set similarity; canonical-address character and token-set similarity.
- Ten configured channel indicators.

No country, entity-ID value or final-matcher prediction is a feature. IDs set only reproducible partition membership
and stable ties. The generic country-scoped blocking keys are unchanged, and there are no country-specific rules. The largest
gains are cheap score, address token-set similarity, name character similarity, number state and name token-set
similarity.

For inference, canonical name/address strings are computed once into `candidate-text-v1`. This is an mmap-backed UTF-8
blob with one int64 offset per record, tied to the raw-source hash and normalization version. Raw and cached text
parity is tested; the cache changes runtime, not feature values.

## 10. Training / held-out split

The query split hashes the Source 1 ID with salt `phase9-candidate-ranker-v1` and training fraction 0.7. It does not
depend on candidate count or target source. **28,035 training queries, 12,135 held-out queries.** An explicit shard
audit confirmed that no held-out query enters training and that all **95,873 reachable training positives** are kept.
The held-out side has **41,642 reachable positives**.

## 11. Training size

**2,792,408 rows**: 95,873 positives and 2,696,535 negatives. Per query and source, sampling keeps all positives, up
to 30 reproducible random candidates and up to 20 candidates with the strongest retrieval rarity, deduplicated.
Evaluation scores the complete broad pool; evaluation negatives are not sampled. Scores are used only for ranking,
with no calibration or score threshold. Collection took 1,166.53 s at 5,355.24 MiB peak RSS.

## 12. Ranker model artifact

One configuration: LightGBM binary objective, 180 trees, 31 leaves, learning rate 0.07, minimum leaf population 60,
L2 regularization 2, 3 threads, deterministic column-wise training, seed 901. Fitting took **37.98 s** at 885.65 MiB.

- `output/candidate_ranker_v1.txt` — LightGBM text model, 643,571 bytes (628.5 KiB), SHA256
  `13d4d8415124e67539fcd595ce1e3e74de8cf588b2ad1b2eddc03638b777fc1e`. It is committed with `git add -f`, as the
  Phase 6 model was.
- Training metadata: `data/interim/candidates/phase9_ranker_model.json` and `phase9_training_data.json` (hashed in
  the frozen policy).
- Software: Python 3.14.7, NumPy 2.5.3, LightGBM 4.7.0.

## 13. Top-K / rescue-budget results

The budget is the number of ranker rescues added **per query and target source**, chosen from broad candidates
outside Phase 8. Phase 8 candidates are never removed. The sample evaluation took 4,415 s for 40,170 queries and 282.6M
scored pairs.

Held-out (12,135 queries, 42,044 links):

| Budget | Recall | Oracle | Mean | Median | p95 | p99 | Max | Added links | Added pairs | Links / 1,000 added pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (Phase 8) | 0.96114 | 0.98584 | 42.49 | 39 | 70 | 87 | 123 | — | — | — |
| 1 | 0.97674 | 0.99198 | 44.49 | 41 | 72 | 89 | 125 | 656 | 24,270 | 27.03 |
| 2 | 0.98050 | 0.99327 | 46.49 | 43 | 74 | 91 | 127 | 814 | 48,540 | 16.77 |
| 3 | 0.98242 | 0.99392 | 48.49 | 45 | 76 | 93 | 129 | 895 | 72,810 | 12.29 |
| 5 | 0.98478 | 0.99477 | 52.49 | 49 | 80 | 97 | 133 | 994 | 121,350 | 8.19 |
| **8** | **0.98644** | **0.99538** | **58.49** | **55** | **86** | **103** | **139** | **1,064** | **194,158** | **5.48** |
| 10 | 0.98694 | 0.99560 | 62.49 | 59 | 90 | 107 | 143 | 1,085 | 242,696 | 4.47 |

Standalone ranking quality: over the whole broad pool, the ranker's held-out top 8 per query and source contain 97.73% (S2)
and 97.82% (S3) of reachable positives. Its top 1 contains 49.05% and 46.65%.

## 14. Sample vs held-out comparison

| Budget | Sample recall | Held-out recall | Sample oracle | Held-out oracle | Sample mean | Held-out mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.961081 | 0.961136 | 0.985977 | 0.985836 | 42.44 | 42.49 |
| 1 | 0.976817 | 0.976739 | 0.992197 | 0.991985 | 44.44 | 44.49 |
| 2 | 0.980699 | 0.980497 | 0.993488 | 0.993273 | 46.44 | 46.49 |
| 3 | 0.982586 | 0.982423 | 0.994114 | 0.993923 | 48.44 | 48.49 |
| 5 | 0.985013 | 0.984778 | 0.994955 | 0.994770 | 52.44 | 52.49 |
| **8** | **0.986561** | **0.986443** | **0.995512** | **0.995382** | **58.44** | **58.49** |
| 10 | 0.987231 | 0.986942 | 0.995711 | 0.995601 | 62.44 | 62.49 |

At +8, the training-partition queries reach 0.98661 / 0.99557 and held-out 0.98644 / 0.99538. The gaps
(0.00017 recall, 0.00019 oracle) are about the same as the Phase 8 baseline difference between partitions, so the
ranker shows little observable overfit. At +8 the sample distribution is median 55, p90 78, p95 86, p99 103, max 139, with
zero empty queries. S2 recall is 0.98624, S3 0.98663, India 0.97962 and US 0.99093 (held-out).

## 15. Why +8 was selected

+8 is the smallest budget that clears both primary candidate-ceiling gates on held-out queries (recall ≥ 98.5%,
oracle ≥ 0.995) while staying under the ~60 candidates/query ceiling: 98.644% / 0.99538 / 58.49. +5 misses both
gates (98.478% / 0.99477). +10 adds only 21 held-out links (+0.0005 recall, +0.0002 oracle) for 48,538 more pairs
(0.43 links per 1,000) and exceeds the 60/query ceiling at 62.49. The decision is frozen and was not retuned.

## 16. Frozen blocking-v4 semantics

`experiments/configs/blocking_v4.json` (SHA256 `e4916142ccf3aaa96fc1266068e03d5c66d8e4cf0aefabaf9fe07b900b681347`,
`status: frozen`):

- Candidate set per query and target source = the Phase 8 blocking-v3 final set, plus the 8 best-ranked broad candidates
  outside it. The ranker scores every broad candidate (deep pool + overflow ≤ 1000). Only non-Phase-8 candidates are
  ranked, by descending ranker score, then ascending minimum shared frequency, then binary target-ID order.
- `iter_ranked_candidates` (`src/matching/candidate_ranker.py`) is the only implementation. It asserts that broad
  retrieval contains every Phase 8 pair, keeps Phase 8 provenance flags (1/2/4/8), and marks rescues with flag
  16. Batch size 100; disk reserve 10.5 GiB.
- The policy hashes 49 files and 2 directory trees. `load_frozen` re-verifies them before any frozen run. Key hashes:

| Artifact | SHA256 |
| --- | --- |
| `experiments/configs/blocking_v2.json` | `ad6ccfd69266860a4ec5019894fd8da73499fec2ef454f084accb84e84334d04` |
| `experiments/configs/blocking_v3.json` | `42ab3b48abfacb758d77ed9172d68791832a4cb9ed649da76fe00250c8c9ca65` |
| `experiments/configs/matcher_v1.json` | `5301cf2d22f170de18d7f6bf1f08c603bf157dfa8ad4ca5cc27fc1dbb489ad44` |
| `experiments/configs/candidate_ranker_v1.json` | `162482fb310eb4a60773ba093a04fc23d28db3b04ef4b39ac7144edb6eb20c34` |
| `experiments/configs/validation.json` | `5b20ad320cbc565c896f09ca27ce0d7c713985cdf086b1fad604b7ea2b18d056` |
| `output/candidate_ranker_v1.txt` | `13d4d8415124e67539fcd595ce1e3e74de8cf588b2ad1b2eddc03638b777fc1e` |
| `src/matching/candidate_ranker.py` | `afa196da1775d36dc15be2dc20525d1f849e1d77beea70852a045add0abf1f9f` |
| `src/matching/candidate_text.py` | `f6c6e3fb681e83380769260c70ad9cf2d119bb4eebfeed9519a5b571d5c0a6a5` |
| `src/blocking/overflow.py` | `1759d5ec1c2edfc5e5340b35d99261199fae654f7b961a4b81b51ae339c457ce` |
| `src/matching/final_candidates.py` | `4c36e65db79d7d95d0f891086d31b584afca750a9272ffc41424cd16c292349e` |
| `src/blocking/recovery.py` | `709d34c424e109fa8ad038556d483b8e93352c3f5693a391f9f5725a0fd17cd9` |
| `data/interim/candidates/phase9_training_data.json` | `2f4a4e652a9066f28c2ded07e49ec4d7618152f1df7c6a6cefdfcd55df0dc903` |
| `data/interim/candidates/phase9_ranker_model.json` | `1e013c4e7bd835882d1ca8dda6f76fc7c08f22d9cf9427cb5af80247c528c172` |
| `data/interim/validation/metadata.json` (split, fingerprint `1dd67d77…3645`) | `d1f88d2fe08fc7698044d461a3ab5782ba807c2602f57131d096aca84a44f874` |
| `data/interim/blocking/overflow-v1/` (tree hash) | `7e9839559821b43a94650ff723d87b6ae4ae2f1d1a6b6a24f8f324883da012be` |
| `data/interim/blocking/candidate-text-v1/` (tree hash) | `7c10f1c51baad3f238d949146c4a766eb707678cc2f43e03fe0c6d8fcbfb27e7` |

The policy also hashes the channel, candidate, index, ranking, store, pairwise-feature and rule modules, plus the
store, base-index and feature-text `meta.json` files.

**Verification against the sample result (no re-run of the 282M-pair evaluation).** The Phase 9 code predates the
evaluation start (15:53), and the model/config hashes equal those in the evaluation and training metadata. The
production `iter_ranked_candidates` was run on a deliberately unaligned 2,000-query sample slice (positions
20,050–22,050). It reproduced the stored +8 selection **pair for pair**: S2 59,424/59,424 and S3 57,030/57,030, with
exact rescue provenance and full Phase 8 preservation. So the sample numbers correspond to the current
code and artifacts, and selection does not depend on batch or shard boundaries. A two-shard, spawn-worker smoke test of
the frozen runner matched per-query pairs, Phase 8 pairs, found links and Phase 8 found links exactly.

## 17. Full-development confirmation — started, then terminated (deadline)

The frozen runner (`scripts/run_large_bucket_recovery.py --evaluate`) was extended to split queries into contiguous
shards across spawn worker processes. Each shard persists its per-query counts, so an interrupted run of the same
policy resumes. The Phase 8 subset must reproduce the Phase 8 report exactly. The confirmation rule was pre-registered in
the policy: recall ≥ held-out − 0.003, oracle ≥ held-out − 0.001, and |mean − held-out| ≤ 2.

Measured steady throughput is ~6.1 s per 100 query-sources in one process: features ~4.4 s, prediction ~1.0 s,
retrieval ~0.8 s, plus the Phase 8 lockstep. Full development is therefore ~67 h serial, and ≈ 17–22 h with four
workers, which is memory-bound (~1.1 GiB private per worker).

The run started at 18:09 with 4 workers and 199 shards of ~10,000 queries. After the deadline change it was
**terminated at ~18:23 during warm-up, before any shard completed**. No development metrics were computed or
inspected. `data/interim/candidates/blocking-v4_development_report.started` and the empty shard directory record the
aborted run. Its minimum available memory was 2.97 GiB (the browser was using swap), and free disk stayed ≥ 14.5 GiB.

**blocking-v4 has no full-development confirmation.**

## 18. Frozen validation — not run (deadline)

Validation has 220,410 queries, or 440,820 query-sources: ≈ 26,900 s serial, and **≥ 1.9 h** even with perfect 4-worker
scaling. That is above the 60-minute limit, so the run was skipped under the deadline rule. The runner also refuses validation
without a confirming development report. Validation labels were not read for blocking-v4. **blocking-v4 has no
validation result.**

## 19–21. Runtime, peak RAM, minimum free disk

| Step | Runtime | Peak RSS (incl. mapped files) |
| --- | ---: | ---: |
| Overflow build + first ceiling pass | 157.82 s build | 4,008 MiB |
| Cached-index ceiling pass (40,170 queries) | 128.55 s | 2,983 MiB |
| Training collection | 1,166.53 s | 5,355 MiB |
| Ranker fit | 37.98 s | 886 MiB |
| Sample/held-out budget evaluation (282.6M pairs) | 4,415.18 s | 5,265 MiB |
| Parity verification (2,000 queries × 2 sources) | 309.9 s | ~4.9 GiB (≈ 1.1 GiB private) |
| Aborted full-development run | ~14 min, no shard done | 4 workers ≈ 1.3 GiB private each |

The minimum free disk observed across Phase 9 was **14.52 GiB** (sample evaluation and text-cache build), always
above the 10.5 GiB reserve. Core dumps are disabled in every heavy runner.

## 22. Remaining candidate loss at +8

| Partition | Links | Missed | Unreachable by broad retrieval | Reachable, not selected |
| --- | ---: | ---: | ---: | ---: |
| Sample | 138,853 | 1,866 | 1,338 (1,306 bucket > 1000, 32 no shared key) | 528 |
| Held-out | 42,044 | 570 | 402 | 168 |

On the sample, India has 1,142 missed links (722 unreachable, 420 not selected) and US 724 (616 / 108). S2 has 921
and S3 945. The next recall step would be buckets above 1,000, which are mostly address-only evidence with missing or
conflicting numbers, or more rescue slots for Indian queries. Neither is pursued here.

## 23. Official candidate-set semantics

For blocking-v4, the set yielded by `iter_ranked_candidates` (Phase 8 pairs plus the ≤ 8 rescues per query and source)
would be the **exact set the final matcher scores and that `output/candidate_pairs.tsv` must contain**. The
282.60M-pair broad pool is temporary retrieval only. The candidate ranker is a separate filtering model; the
Phase 6 final matcher was neither called nor retrained. No candidate TSV, matching results or submission was
generated in this phase. `output/candidate_pairs.tsv` is still the Phase 5 test set (SHA256 `bd0cd868…7394`).

## 24. Recommendation for the FINAL MATCHER phase

1. **Use Phase 8 blocking-v3 as the official candidate set for the deadline submission**
   (`experiments/configs/blocking_v3.json`, `iter_final_candidates`). It is the only candidate policy with a frozen
   validation result (recall 0.96101, oracle 0.98606, mean 42.48). Test Source 1 has 1,732,544 queries, and
   blocking-v3 test generation should take ~2 h with 3 workers (development: 8,168 s for 1,986,411 queries).
   Write exactly that set to `output/candidate_pairs.tsv`.
2. **Retrain the final matcher on blocking-v3 candidates** (development partition, frozen split), with the Phase 8
   provenance flags available to it. Tune the single global threshold for Macro F0.5 on development, then run
   one validation, test inference, the official validator and submission. Phase 6 was trained on Phase 5 candidates,
   and blocking-v3 adds conflict/rank/deep rescues it has not seen.
3. **Do not use blocking-v4 for the deadline submission.** At ~6.1 s per 100 query-sources, blocking-v4 test
   generation would take ~59 h serial and ≈ 15–20 h with 4 workers. It would also need test overflow indexes and
   test text caches, and it is unconfirmed on full development and validation. Its rescues are low precision
   (5.5 true links per 1,000 added pairs), so a matcher would need retraining on them to benefit.
4. **After the deadline**, blocking-v4 can be confirmed with
   `python scripts/run_large_bucket_recovery.py --evaluate --partition development --resume` (same frozen policy,
   pre-registered gate), then one validation. If it confirms, retrain the matcher on the sample's v4 candidates,
   which can be rebuilt from `phase9_phase8_pairs_s*.npy` and `phase9_ranker_rescues.npy` without rescoring.

## Reproduction

Use the repository virtual environment (`source .venv/bin/activate`).

```bash
python scripts/analyze_large_buckets.py
python scripts/run_large_bucket_recovery.py --ceiling
python scripts/train_candidate_ranker.py collect
python scripts/train_candidate_ranker.py train
python scripts/train_candidate_ranker.py evaluate
# frozen confirmation (not completed in Phase 9); validation requires a confirming development report
python scripts/run_large_bucket_recovery.py --evaluate --partition development --workers 4 --shard-queries 10000 --resume
python -m pytest -q
git diff --check
```

Training collection refuses to overwrite existing shards. Generated indexes, text caches, shards, candidate traces
and JSON diagnostics stay under the ignored `data/interim/`.
