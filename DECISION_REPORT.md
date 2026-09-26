# Phase 7 — Decision optimization over frozen Phase 6 probabilities

## Objective and frozen starting point

Test whether simple query-aware decisions improve the frozen LightGBM model over
its global threshold of **0.660**. Work starts at
`5ae7881616c013259467d0d60094ca62d725946a` on `phase7-decision`.

The model remains `output/supervised_v1/model.txt`, SHA256
`744aa05566e3d77f0c10c9d01a450a8bf518fd3142b97efe87629b41d52b9e75`.
Its metadata SHA256 is
`0fcab02ccc5b8b186efc85e8608e98cf449ebda6aa3888223bd73e8005d6029d`.
The loader verifies these hashes, the exact 42-feature order, upstream config
hashes, original split fingerprint, LightGBM version, and frozen threshold.

Candidate generation, normalization, features, model parameters, model artifact,
Phase 5 prefilter, candidate-pair semantics, and frozen validation split are
unchanged. No retraining, new features, external data, country-based decisions,
test inference, leaderboard optimization/submission, or Phase 8 work is included.

Only the original **7,994 Phase 6 tune queries** are scored for development.
They were excluded from model fitting. An optional query filter in the existing
Phase 6 collector applies the original tune hash before ground-truth selection
and feature extraction. Existing callers retain their original behavior; a
filter is forbidden for frozen validation. Feature batches are scored immediately,
then discarded. Only compact scores and existing candidate metadata remain in
RAM; no feature matrix or probability cache is persisted.

Before any search, the complete tune baseline must exactly reproduce the saved
Phase 6 diagnostics, including Macro F0.5 **0.9259311292573912**, **198,913**
screened pairs, query membership, candidate statistics, and country/source metrics.
Any discrepancy stops the run before decision search.

## Fixed search and selection protocol

Configuration: `experiments/configs/decision_v1.json`.

- One query-level split using SHA-256 through the existing `in_sample` helper:
  salt `decision-v1-confirm-split`, confirm fraction 0.40. Every candidate follows
  its query; no country is used in splitting or decisions.
- **121 configurations total**, fixed before inspecting results. No refinement,
  adaptive grids, source-specific thresholds, or combined relative/margin plus
  abstention policies.
- Global: 26 thresholds from 0.55 through 0.80, step 0.01.
- Relative-to-best: T in [0.60, 0.64, 0.66, 0.68, 0.72, 0.76], alpha in
  [0.50, 0.60, 0.70, 0.80, 0.90, 0.95] (36 configurations).
- Margin-from-best: the same six T values, delta in
  [0.02, 0.05, 0.10, 0.15, 0.20] (30 configurations).
- Single-prediction abstention: a global-threshold base with the same six T
  values, S in [0.65, 0.70, 0.75, 0.80, 0.85, 0.90], only S > T (29 configurations).
  Exactly one surviving match must satisfy S; multiple survivors remain unchanged.
- Best score and survivor counts span **both S2 and S3 within each query**.
  Threshold comparisons are inclusive, ties retain all qualifying matches,
  and zero predictions are always allowed.
- Each family chooses one winner using decision-search only. All four winners
  are fixed before their confirm scores are computed. A winner is eligible only
  if gains on both search and confirm are strictly positive and safety checks pass.
- Predeclared safety checks, on both partitions: at most 0.005 absolute micro
  precision loss and at most 0.02 absolute singleton-correct-rate loss. Country
  diagnostics do not enter selection.
- Select the eligible winner with higher confirm Macro F0.5, then fewer FP,
  simpler policy (global, relative/margin, abstention), then fewer predictions
  and stricter parameters. Stable grid order resolves any remaining tie.
  If none qualify, retain the Phase 6 policy.
- Save the chosen policy/config artifact and its SHA256 **before** evaluating it
  on the complete tune set. The full-tune score is verified with both the canonical
  evaluator and the existing fast Macro F0.5 implementation.
- Admit validation only for complete-tune gain >=0.002, strictly positive confirm
  gain, and the same precision/singleton safety checks. Otherwise retain Phase 6
  in production and do not load validation labels.
- An admitted validation uses the frozen policy once, reproduces Phase 6's
  validation baseline on the same scores, and compares the selected decisions.
  An exclusive start marker and report guard prohibit another run; no force flag.

The Phase 6 threshold was originally selected on all 7,994 tune queries; these
queries are out of sample for model fitting, but not for that historical threshold
selection. Decision-confirm isolates the new parameter search, while selection
among the four family winners still uses confirm. It is not an untouched final
test set; the separate frozen validation is reserved for policies passing the gate.

## Resources and reproduction

The existing three feature workers, 1,000-query batches, and six-thread model
scoring path are reused. Disk and available system memory are checked during
streaming and around major stages, with stop guards of 10.5 GiB free disk and
1,024 MiB available memory. Core dumps are disabled. Protected model, metadata,
and Phase 5 submission hashes are verified before and after the run.

```bash
python scripts/run_decision.py --development
python scripts/run_decision.py --validation  # only when the Phase 7 gate passes
python -m pytest -q
```

Development refuses an existing report or frozen policy artifact. Detailed
search-only diagnostics for all 121 configurations and search/confirm diagnostics
for the four winners are stored in the small ignored development JSON report under
`data/interim/matching/`. The compact policy/config artifact is
`output/decision_v1/policy.json`. It is a candidate policy; production adoption is
controlled by the gate recorded in the development report.

## Outcome: gate failed; retain Phase 6

**Phase 7 is complete and unsuccessful under the required material-improvement gate.**
The best robust candidate is relative-to-best: `score >= 0.660` and `score >= 0.70 * query_best`. Its complete-tune score is **0.9261902788016474**, versus **0.9259311292573912**, a gain of **+0.00025914954425620973**. Confirm gain is **+0.00018917308338517458**. This is below the required **+0.002** complete-tune improvement.

**Production retains the Phase 6 global threshold 0.660. No Phase 7 validation was run or started, and no validation labels were selected for this phase.** The historical Phase 6 frozen validation score remains 0.9253015854145195; it is not a Phase 7 result. No policy was altered after confirm or complete-tune evaluation.

## Exact baseline reproduction and split

All saved Phase 6 full-tune diagnostics reproduced exactly, not merely within tolerance: Macro F0.5, TP/FP/FN, precision/recall, singleton counts, prediction multiplicities, exact matches, candidate distributions/recalls/oracles, and S2/S3 plus India/US diagnostics. Model and metadata hashes, feature order, threshold, and tune membership also matched.

Full-tune membership SHA256: `55831052fb4487447642c18ceb032fc3c53045ab3c7503a3a11706629c88aeb8`.

| Partition | Queries | Pairs | Positives | Negatives |
| --- | --- | --- | --- | --- |
| search | 4788 | 119500 | 15645 | 103855 |
| confirm | 3206 | 79413 | 10460 | 68953 |

- Search membership SHA256: `23d6962841ea608ed2098036bb3f6706a6d40c3261224915df057af042365a53`.
- Confirm membership SHA256: `05434c42a03e72db41463d23d6698111cedd92c0b17e8d8736a18345d0e14a95`.

| Stage | Pairs | Mean/query | Median | p95 | Max | Recall | Oracle Macro F0.5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Phase 3B | 687721 | 86.029647235427 | 100 | 100 | 100 | 0.949473835952 | 0.979933266571 |
| Phase 5 screen | 198913 | 24.882787090318 | 21 | 52 | 100 | 0.940788525299 | 0.976825767065 |

The unchanged screen supplied all 198,913 decision candidates. Every prediction remains a subset of that set; candidate recall and oracle are identical for every policy.

## Family winners and rejection decisions

| Family | Frozen search-selected parameters | Search Macro F0.5 | Search gain | Confirm Macro F0.5 | Confirm gain | Outcome |
| --- | --- | --- | --- | --- | --- | --- |
| global | threshold=0.66 | 0.926994398206 | +0.000000000000 | 0.924343190478 | +0.000000000000 | Baseline retained; no gain |
| relative | alpha=0.7, threshold=0.66 | 0.927300403334 | +0.000306005128 | 0.924532363561 | +0.000189173083 | Robust, but full-tune gate failed |
| margin | delta=0.2, threshold=0.66 | 0.925448612138 | -0.001545786068 | 0.922852436863 | -0.001490753614 | Rejected: negative search and confirm gains |
| singleton | single_threshold=0.7, threshold=0.64 | 0.926875495178 | -0.000118903029 | 0.923867813416 | -0.000475377061 | Rejected: negative search and confirm gains |

The relative policy was the only family winner with positive search and confirm gains. Precision/singleton safeguards passed for it on both partitions and complete tune. Margin and abstention were rejected rather than expanded into new variants. The best global threshold remained 0.660.

## Search and confirm diagnostics for each family winner

| Partition | Policy | TP | FP | FN | Precision | Recall | Correct singletons / total | Singleton false merges | Zero / one / multiple | Exact matches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| search | Phase 6 | 14468 | 420 | 2176 | 0.971789361 | 0.869262197 | 260 / 289 | 29 | 333 / 469 / 3986 | 2977 |
| search | global | 14468 | 420 | 2176 | 0.971789361 | 0.869262197 | 260 / 289 | 29 | 333 / 469 / 3986 | 2977 |
| search | relative | 14383 | 377 | 2261 | 0.974457995 | 0.864155251 | 260 / 289 | 29 | 333 / 477 / 3978 | 2952 |
| search | margin | 14091 | 283 | 2553 | 0.980311674 | 0.846611391 | 260 / 289 | 29 | 333 / 536 / 3919 | 2836 |
| search | singleton | 14512 | 434 | 2132 | 0.970962130 | 0.871905792 | 263 / 289 | 26 | 342 / 447 / 3999 | 3003 |
| confirm | Phase 6 | 9658 | 275 | 1446 | 0.972314507 | 0.869776657 | 150 / 177 | 27 | 204 / 339 / 2663 | 2023 |
| confirm | global | 9658 | 275 | 1446 | 0.972314507 | 0.869776657 | 150 / 177 | 27 | 204 / 339 / 2663 | 2023 |
| confirm | relative | 9617 | 253 | 1487 | 0.974366768 | 0.866084294 | 150 / 177 | 27 | 204 / 347 / 2655 | 2010 |
| confirm | margin | 9376 | 182 | 1728 | 0.980958359 | 0.844380403 | 150 / 177 | 27 | 204 / 386 / 2616 | 1909 |
| confirm | singleton | 9685 | 291 | 1419 | 0.970829992 | 0.872208213 | 151 / 177 | 26 | 207 / 332 / 2667 | 2033 |

Baseline partition Macro F0.5: search **0.9269943982062449**, confirm **0.9243431904778805**. Winner scores and deltas are in the preceding table. All 121 search configurations have the same detailed metrics in the local JSON report; only the four frozen family winners were evaluated on confirm.

### Source diagnostics

| Partition | Policy | Source | TP | FP | FN | Precision | Recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| search | Phase 6 | source2 | 6973 | 209 | 1040 | 0.970899471 | 0.870210907 |
| search | Phase 6 | source3 | 7495 | 211 | 1136 | 0.972618739 | 0.868381416 |
| search | global | source2 | 6973 | 209 | 1040 | 0.970899471 | 0.870210907 |
| search | global | source3 | 7495 | 211 | 1136 | 0.972618739 | 0.868381416 |
| search | relative | source2 | 6930 | 191 | 1083 | 0.973177924 | 0.864844627 |
| search | relative | source3 | 7453 | 186 | 1178 | 0.975651263 | 0.863515236 |
| search | margin | source2 | 6777 | 142 | 1236 | 0.979476803 | 0.845750655 |
| search | margin | source3 | 7314 | 141 | 1317 | 0.981086519 | 0.847410497 |
| search | singleton | source2 | 6990 | 214 | 1023 | 0.970294281 | 0.872332460 |
| search | singleton | source3 | 7522 | 220 | 1109 | 0.971583570 | 0.871509674 |
| confirm | Phase 6 | source2 | 4666 | 146 | 681 | 0.969659185 | 0.872638863 |
| confirm | Phase 6 | source3 | 4992 | 129 | 765 | 0.974809607 | 0.867118291 |
| confirm | global | source2 | 4666 | 146 | 681 | 0.969659185 | 0.872638863 |
| confirm | global | source3 | 4992 | 129 | 765 | 0.974809607 | 0.867118291 |
| confirm | relative | source2 | 4650 | 132 | 697 | 0.972396487 | 0.869646531 |
| confirm | relative | source3 | 4967 | 121 | 790 | 0.976218553 | 0.862775751 |
| confirm | margin | source2 | 4535 | 100 | 812 | 0.978425027 | 0.848139143 |
| confirm | margin | source3 | 4841 | 82 | 916 | 0.983343490 | 0.840889352 |
| confirm | singleton | source2 | 4674 | 154 | 673 | 0.968102734 | 0.874135029 |
| confirm | singleton | source3 | 5011 | 137 | 746 | 0.973387723 | 0.870418621 |

### Country diagnostics

| Partition | Policy | Country | Macro F0.5 | Precision | Recall | TP / FP / FN | Correct singletons / total | Singleton false merges |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| search | Phase 6 | India | 0.893348650 | 0.957799820 | 0.827903352 | 5311 / 234 / 1104 | 93 / 109 | 16 |
| search | Phase 6 | US | 0.947957410 | 0.980092048 | 0.895199922 | 9157 / 186 / 1072 | 167 / 180 | 13 |
| search | global | India | 0.893348650 | 0.957799820 | 0.827903352 | 5311 / 234 / 1104 | 93 / 109 | 16 |
| search | global | US | 0.947957410 | 0.980092048 | 0.895199922 | 9157 / 186 / 1072 | 167 / 180 | 13 |
| search | relative | India | 0.893583511 | 0.961489323 | 0.821200312 | 5268 / 211 / 1147 | 93 / 109 | 16 |
| search | relative | US | 0.948307742 | 0.982113996 | 0.891093949 | 9115 / 166 / 1114 | 167 / 180 | 13 |
| search | margin | India | 0.890978231 | 0.970404098 | 0.797349961 | 5115 / 156 / 1300 | 93 / 109 | 16 |
| search | margin | US | 0.946925412 | 0.986048555 | 0.877505132 | 8976 / 127 / 1253 | 167 / 180 | 13 |
| search | singleton | India | 0.892339830 | 0.956069571 | 0.831176929 | 5332 / 245 / 1083 | 95 / 109 | 14 |
| search | singleton | US | 0.948392971 | 0.979827089 | 0.897448431 | 9180 / 189 / 1049 | 168 / 180 | 12 |
| confirm | Phase 6 | India | 0.892248483 | 0.961344983 | 0.826166098 | 3631 / 146 / 764 | 47 / 61 | 14 |
| confirm | Phase 6 | US | 0.945397053 | 0.979044834 | 0.898345506 | 6027 / 129 / 682 | 103 / 116 | 13 |
| confirm | global | India | 0.892248483 | 0.961344983 | 0.826166098 | 3631 / 146 / 764 | 47 / 61 | 14 |
| confirm | global | US | 0.945397053 | 0.979044834 | 0.898345506 | 6027 / 129 / 682 | 103 / 116 | 13 |
| confirm | relative | India | 0.892380598 | 0.964171123 | 0.820477816 | 3606 / 134 / 789 | 47 / 61 | 14 |
| confirm | relative | US | 0.945623656 | 0.980587276 | 0.895960650 | 6011 / 119 / 698 | 103 / 116 | 13 |
| confirm | margin | India | 0.890189405 | 0.972874720 | 0.791581342 | 3479 / 97 / 916 | 47 / 61 | 14 |
| confirm | margin | US | 0.944279116 | 0.985790705 | 0.878968550 | 5897 / 85 / 812 | 103 / 116 | 13 |
| confirm | singleton | India | 0.891477808 | 0.959242703 | 0.830034130 | 3648 / 155 / 747 | 48 / 61 | 13 |
| confirm | singleton | US | 0.945115390 | 0.977968573 | 0.899836041 | 6037 / 136 / 672 | 103 / 116 | 13 |

These country diagnostics are descriptive only. They did not affect policy selection, thresholds, or either split.

## Frozen candidate on complete tune

Candidate policy/config artifact SHA256: `3360d6dd30a0bb6f50ed377a37995d8a38e4d68a800db201ffbf57c9ef884680`. The artifact was saved before complete-tune evaluation. Its presence does not imply adoption: the recorded gate failed and production remains Phase 6.

| Metric | Phase 6 retained | Relative candidate |
| --- | --- | --- |
| Macro F0.5 | 0.9259311292573912 | 0.9261902788016474 |
| TP | 24126 | 24000 |
| FP | 695 | 630 |
| FN | 3622 | 3748 |
| Micro precision | 0.9719995165384151 | 0.97442143727162 |
| Micro recall | 0.8694680697707943 | 0.8649272019605017 |
| Correct singletons | 410 | 410 |
| Singleton false merges | 56 | 56 |
| Zero predictions | 537 | 537 |
| One prediction | 808 | 824 |
| Multiple predictions | 6649 | 6633 |
| Exact-match entities | 5000 | 4962 |

The candidate removes 65 false positives but also 126 true positives. Precision rises by 0.2422 percentage points, recall falls by 0.4541 points, and singleton errors are unchanged. Exact matches decrease by 38. The per-query Macro F0.5 improvement is too small to justify adoption under this protocol.

| Matcher | Source | TP | FP | FN | Precision | Recall |
| --- | --- | --- | --- | --- | --- | --- |
| Phase 6 | source2 | 11639 | 355 | 1721 | 0.9704018676 | 0.8711826347 |
| Phase 6 | source3 | 12487 | 340 | 1901 | 0.9734934123 | 0.8678760078 |
| Relative candidate | source2 | 11580 | 323 | 1780 | 0.9728639839 | 0.8667664671 |
| Relative candidate | source3 | 12420 | 307 | 1968 | 0.9758780545 | 0.8632193495 |

| Matcher | Country | Macro F0.5 | Precision | Recall | Correct singletons / total | Singleton false merges |
| --- | --- | --- | --- | --- | --- | --- |
| Phase 6 | India | 0.892899096323 | 0.9592362154 | 0.8271970398 | 140 / 170 | 30 |
| Phase 6 | US | 0.946942909519 | 0.9796761081 | 0.8964458614 | 270 / 296 | 26 |
| Relative candidate | India | 0.893091972858 | 0.9625772860 | 0.8209065680 | 140 / 170 | 30 |
| Relative candidate | US | 0.947244215534 | 0.9815067160 | 0.8930216082 | 270 / 296 | 26 |

## Resource measurements and preservation

Development runtime: **167.42 s**. Streaming completed before decision-search; the fixed search itself took about 42 seconds. Peak main-process RSS: **4004.31 MiB**. Minimum observed system available memory: **6057.97 MiB**. Minimum observed free disk: **16.945484 GiB**. These timings exclude tests.
`RUSAGE_CHILDREN` reported zero under the Python forkserver and does not measure the feature workers here. Worker/aggregate peak RSS is unavailable; the main-process RSS and system-memory snapshots are reported without presenting them as an aggregate peak.

| Checkpoint | Free disk GiB |
| --- | --- |
| start | 16.984764 |
| streaming_scores_complete | 16.981850 |
| before_decision_search | 16.981850 |
| after_decision_search | 16.945492 |
| decision_gate_complete | 16.945484 |
| finish | 16.945484 |

No probability/feature caches or raw-data copies were created. The detailed search report is 976,825 bytes and remains an ignored diagnostic file; it is not a matrix. Only the compact frozen policy/config artifact is included with the code and Markdown report.

- Unchanged `output/candidate_pairs.tsv`: `bd0cd868dfaa017e047637994ef38fa107d4c28e50e63f575d7512fabcb77394`.
- Unchanged `output/matching_results.tsv`: `4cb9a2327cd6cecb6a131dba5b7756a2c4d8084cfa212dc7a6efe383fc1cdaee`.
- Unchanged `output/supervised_v1/metadata.json`: `0fcab02ccc5b8b186efc85e8608e98cf449ebda6aa3888223bd73e8005d6029d`.
- Unchanged `output/supervised_v1/model.txt`: `744aa05566e3d77f0c10c9d01a450a8bf518fd3142b97efe87629b41d52b9e75`.

## Limitations and stopping condition

This bounded experiment does not prove all possible decision rules are ineffective. It answers the requested question for four simple predeclared families without extending the search after negative results. The observed gain is too small, and no new validation result is available or claimed.

**Phase 7 does NOT address the current ~0.9775 candidate oracle ceiling. Candidate recall remains the next major bottleneck; candidate-recall improvement is deferred to Phase 8.** No candidate-recall research or Phase 8 implementation was started. No raw test records were read, and there was no France-specific behavior, test inference, submission generation, or leaderboard tuning. The phase stopped when the development gate failed.

## Final verification

`python -m pytest -q`: **237 passed** in 28.77 s after the failed gate, including
all 218 existing tests and 19 new decision tests. The existing harmless
scikit-learn feature-name warning comes from the Phase 6 synthetic probability
comparison. The new tests cover Phase 6 threshold parity, query-local maxima,
relative and margin rules, abstention, zero/multiple matches, ties, candidate
subsets, split determinism, canonical metric parity, confirm-label isolation,
open country labels, selection safeguards, and frozen-validation rerun protection.
`git diff --check` passed. There was one development run and no Phase 7 validation
attempt; no further optimization followed the outcome.
