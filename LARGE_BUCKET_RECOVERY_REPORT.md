# Phase 9 — Large-bucket recovery and candidate ranker

Candidate generation only. Starting branch: `phase9-large-bucket-recovery`; starting commit: `5aa27474a9a18f4daa8324f0977591b8fcf2c816`. All design labels come from the existing 40,170-query development sample. The frozen query split and all Phase 8 configurations/artifacts are preserved. No final matcher training, threshold optimization, test inference, enrichment or submission is part of this phase.

## Phase 8 baseline

| Partition | Recall | Oracle Macro F0.5 | Mean candidates |
| --- | ---: | ---: | ---: |
| Development sample | 0.961081 | 0.985977 | 42.438 |
| Full development | 0.96054 | 0.98592 | 42.49 |
| Frozen validation | 0.96101 | 0.98606 | 42.48 |

The sample has 138,853 true links. Phase 8 recovers 133,449 and misses 5,404: 32 with no shared key; 2,736 truncated by index storage; 1,792 blocked by frequency caps; 96 by key limits; 430 by union ranking; and 318 by the Phase 5 prefilter.

## Existing indexing and storage

`PostingIndex` sorts distinct `(uint64 key, int32 record)` postings. `split_oversized` removes **all postings** of a bucket above `index_keep`, retaining only `large_keys` and `large_counts`. It does not keep the first 200 records. Nine channels keep buckets through 200; `name_x_address_word` keeps only through 50. Query-time frequency caps and rare-key limits are separate restrictions. `deepen` removes key limits and lifts frequency caps to the existing storage limit. Phase 8 then preserves its Phase 5 screen and adds conflict, union-rank and eight deep rescues per source.

Phase 9 stores only the missing oversized buckets, using a unique uint64 key dictionary, int64 CSR offsets and int32 target rows. The base indexes remain unchanged. Target records stream in bounded 10,000-row batches; known bucket counts allocate exact slots, without external-sort staging or duplicate base postings. Files are mmap-backed, versioned `overflow-csr-v1`, source-separated, tied to source input hashes and reused when the identity matches. Incomplete builds are removed on failure; unexpected existing staging directories or stale indexes are rejected.

A preallocation check reserves **10.5 GiB**, and every build/retrieval batch checks free disk again. Heavy runners disable core dumps. Estimated array bytes: **395,042,844**; actual files including headers/metadata: **395,057,544 bytes (376.76 MiB)**. Initial construction took **157.82 seconds**. The measured initial build/ceiling pass peaked at **4,008.08 MiB RSS**.

## Category-B recovery by frequency

Each link is assigned once to its smallest useful shared bucket. Channel ties follow configuration order. The 51–200 row exists because one base index stops at 50. Name similarities below are folded token-set similarities across **all** scripts; Phase 8 masked script mismatches before summarizing name similarity. Address similarities are canonical token-set similarities.

| Best bucket | Unique links | Script mismatch | Median name | Median address | Number missing / agree / partial / conflict |
| --- | ---: | ---: | ---: | ---: | --- |
| 51-200 | 65 | 8 | 0.871 | 0.851 | 32 / 17 / 5 / 11 |
| 201-300 | 371 | 157 | 0.432 | 0.894 | 162 / 141 / 13 / 55 |
| 301-500 | 439 | 216 | 0.176 | 0.904 | 190 / 175 / 19 / 55 |
| 501-750 | 304 | 135 | 0.222 | 0.889 | 147 / 96 / 10 / 51 |
| 751-1000 | 251 | 95 | 0.435 | 0.902 | 133 / 84 / 4 / 30 |
| >1000 | 1,306 | 494 | 0.462 | 0.863 | 729 / 260 / 31 / 286 |

## Overlap-corrected cumulative recovery and pre-build estimates

| Maximum frequency | Additional unique links | Share of Phase 8 misses | Required postings | Array bytes | Projected recall | Projected oracle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 1,430 | 26.46% | 96,911,515 | 395,042,844 | 0.990364 | 0.996872 |
| 300 | 436 | 8.07% | 44,319,312 | 183,062,064 | 0.983205 | 0.994302 |
| 500 | 875 | 16.19% | 66,300,282 | 271,900,952 | 0.986367 | 0.995419 |
| 750 | 1,179 | 21.82% | 83,094,850 | 339,520,792 | 0.988556 | 0.996150 |

Disk estimates added 16 KiB per index for headers and metadata. The 1,000 cap was chosen before training because the 500 cap leaves only 0.000419 oracle margin before compression; the 1,000 cap provides 0.001872 margin at a small storage cost. All eligible overflow keys are retained, rather than selecting keys from positive labels.

At the selected cap, the table below reports attributed links (disjoint), independently reachable links (overlap allowed), and exact postings. Detailed source/channel estimates for all four thresholds are in `data/interim/candidates/phase9_large_bucket_analysis.json`.

| Source / channel | Attributed links | Independent links | Postings | Array MiB |
| --- | ---: | ---: | ---: | ---: |
| S2/address_exact | 0 | 0 | 0 | 0.00 |
| S2/address_number_x_word | 294 | 295 | 6,665,504 | 25.68 |
| S2/address_rare_word | 339 | 344 | 3,159,640 | 12.16 |
| S2/name_exact | 0 | 0 | 8,187 | 0.03 |
| S2/name_folded | 0 | 0 | 8,891 | 0.03 |
| S2/name_folded_compact | 0 | 0 | 10,791 | 0.04 |
| S2/name_rare_token | 18 | 24 | 2,000,339 | 7.69 |
| S2/name_token_pair | 70 | 73 | 3,688,189 | 14.21 |
| S2/name_token_x_address_number | 3 | 6 | 2,437,934 | 9.40 |
| S2/name_x_address_word | 30 | 31 | 30,052,081 | 117.47 |
| S3/address_exact | 0 | 0 | 0 | 0.00 |
| S3/address_number_x_word | 218 | 226 | 6,882,261 | 26.52 |
| S3/address_rare_word | 276 | 281 | 3,023,197 | 11.64 |
| S3/name_exact | 1 | 1 | 8,604 | 0.03 |
| S3/name_folded | 0 | 1 | 9,923 | 0.04 |
| S3/name_folded_compact | 0 | 1 | 11,122 | 0.04 |
| S3/name_rare_token | 24 | 29 | 1,842,804 | 7.09 |
| S3/name_token_pair | 80 | 94 | 3,577,451 | 13.79 |
| S3/name_token_x_address_number | 12 | 44 | 2,662,257 | 10.26 |
| S3/name_x_address_word | 65 | 73 | 30,862,340 | 120.62 |

## Measured broad retrieval ceiling

| Retrieval universe | Pairs | Mean | Recall | Oracle |
| --- | ---: | ---: | ---: | ---: |
| deep | 35,667,649 | 887.92 | 0.980065 | 0.993206 |
| broad | 282,598,169 | 7035.06 | 0.990364 | 0.996872 |

The existing deep pool exactly reproduces Phase 8: 35,667,649 pairs and 136,085 recovered links. Overflow contributes 1,430 further links. The broad universe contains 137,515 true links, **4,066 more than Phase 8**, leaving 1,306 Category-B links above 1,000 and 32 links without a deterministic shared key. This passes the raw-ceiling gate. The temporary universe is never an official candidate set.

The first ceiling pass correctly accumulated link recovery but overwrote per-source candidate counts. The count accumulation was corrected and the cached-index ceiling pass repeated before policy evaluation. The table above uses the corrected combined counts. This was a development-only diagnostic, not a frozen confirmation or validation run.

## Reproduction

Use the repository virtual environment (`source .venv/bin/activate`).

```bash
python scripts/analyze_large_buckets.py
python scripts/run_large_bucket_recovery.py --ceiling
python scripts/train_candidate_ranker.py collect
python scripts/train_candidate_ranker.py train
python scripts/train_candidate_ranker.py evaluate
python -m pytest -q
git diff --check
```

Training collection refuses to overwrite existing shards. Generated indexes, shards, candidate traces and JSON diagnostics remain under ignored `data/interim/`. The small candidate-ranker model is the only generated model intended for version control.

## Candidate-ranker training and features

The query split hashes the Source 1 ID with salt `phase9-candidate-ranker-v1`, training fraction 0.7. It is independent of candidate count and target source. There are **28,035 training queries and 12,135 held-out queries**. An explicit shard audit confirmed that no held-out query enters training and that all **95,873 reachable training positives** are retained. The held-out side contains **41,642 reachable positives**.

The training matrix has **2,792,408 rows**: 95,873 positives and 2,696,535 negatives. Per query/source, sampling keeps all positives, up to 30 reproducible random candidates and up to 20 candidates with the strongest retrieval rarity, deduplicated. This bounds class imbalance and storage. Evaluation expands and scores the complete broad pool; it does not sample evaluation negatives. Scores are used for ranking, without probability calibration or a score threshold.

There is one model configuration: LightGBM binary objective, 180 trees, 31 leaves, learning rate 0.07, minimum leaf population 60, L2 regularization 2, three CPU threads, deterministic column-wise training, seed 901. Model fitting took **37.98 seconds**, peaking at **885.65 MiB RSS**. Collection took **1,166.53 seconds**, peaking at **5,355.24 MiB RSS**. Software: Python 3.14.7, NumPy 2.5.3, LightGBM 4.7.0.

The 34 float32 features are:

- Source 3 indicator.
- Shared name tokens, name Jaccard and name containment; shared address words, address-word Jaccard and containment.
- Number agreement/conflict/missing/partial state; exact folded/compact name and canonical address agreement.
- Minimum shared frequency, summed inverse frequency, key-hit count, channel count and distinct family count.
- Overflow-only provenance and minimum-frequency band.
- Phase 8 cheap score and query-relative cheap-score and rarity ranks.
- Folded-name character and token-set similarity, canonical-address character and token-set similarity.
- Ten configured channel indicators.

No country, entity-ID value or final-matcher prediction is a feature. IDs determine only reproducible partition membership and stable ties. The existing generic country-scoped blocking keys remain unchanged; no country-specific rules are added.

For inference, canonical name/address strings are computed once into `candidate-text-v1`: a UTF-8 blob with one int64 offset per record, mmap-backed and tied to the raw-source hash and normalization version. A bounded string cache decodes requested records. This avoids repeated Python normalization across the broad pool. Raw and cached text parity is tested; this changes runtime, not feature values. It stores derived normalized views, without copying raw TSV files.

## Candidate-set semantics

`iter_ranked_candidates` preserves every Phase 8 candidate and adds up to the configured number of new candidates per query/source. The model ranks only the rescue choices for selection; it never removes a Phase 8 pair. Ties use increasing minimum shared frequency, then binary target-ID order. The yielded `FinalBatch` contains unique pairs in query/target order, with original provenance or the candidate-ranker rescue flag.

This yielded set is the **exact set that a later final matcher must score and that `output/candidate_pairs.tsv` must contain**. The 282.60-million-pair broad sample pool is temporary retrieval. The candidate ranker is a distinct filtering model; the Phase 6 final matcher is neither called nor retrained. No candidate TSV, matching-results TSV or submission is generated in this phase.
