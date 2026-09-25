# Phase 5 — Deterministic Matcher Baseline and First Submission Pipeline

Phase 5 adds the first complete pipeline:

Phase 3B candidates → Phase 4 features → **explicit prefilter** → **deterministic matcher** → `candidate_pairs.tsv` + `matching_results.tsv` → official validator.

There is no learned model: no regression, trees, neural nets or embeddings. Blocking
(`blocking_v2.json`), features (`features_v1.json`), normalization and the frozen
split (`1dd67d77fee74aa0b64477e10f09377ac22854f5eb13780f3aacc4baf94e3645`) are unchanged.

**Validation policy.**
- Every rule and threshold was designed on the 40,170-query development hash sample,
  with development labels only.
- The configuration was then frozen (`experiments/configs/matcher_v1.json`) and run
  **once** on the frozen validation split, with no retuning afterwards. The CLI
  refuses to rerun validation unless `--force` is given.
- Validation queries searched the complete training S2 + S3 universe.
- The full development set (1.99M queries) was not re-processed, and no feature
  matrix was materialized.

## Reproduction

```bash
python scripts/run_matcher.py --sample       # development sample: tuning report and candidate-size Pareto
python scripts/run_matcher.py --validation   # single frozen confirmation (refuses to rerun without --force)
python scripts/run_matcher.py --test         # writes output/candidate_pairs.tsv and output/matching_results.tsv
python scripts/validate_outputs.py           # memory-bounded official validation of both files
python official/utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate /nonexistent --test-dir data/raw/test --check-ids
python -m pytest -q
```

Reports are written to ignored `data/interim/matching/`. The code is in `src/matching/`
(`rules.py`, `pipeline.py`).

## Deterministic matcher

Each candidate's 42 Phase 4 features are reduced to five evidence components:

| Component | Definition | When unavailable (`NaN`) |
| --- | --- | --- |
| Name `N` | `name_token_set_similarity` | Either name empty, **or the two names use different scripts** (`name_script_mismatch`) |
| Address `A` | `address_token_set_similarity` (order-insensitive; Phase 4 showed S3 reorders components) | Either address empty |
| Number state | **agree** (the smaller integer-number set is contained in the other), **partial** (overlap without containment), **conflict** (no shared number), or **unavailable** (a side has no numbers) | — |
| Rarity | `blocking_aggregate_rarity` (Phase 3B evidence) | — |
| Blocking rank | Phase 3B aggregate rank | — |

A candidate is **rule-acceptable** if any of three paths holds. Unavailable evidence
never passes a threshold; it is never read as zero or as agreement.

| Path | Condition |
| --- | --- |
| `name_address` | numbers **agree** and `N ≥ 0.70` and `A ≥ 0.60` |
| `script_mismatch` | numbers **agree**, names in different scripts, and `A ≥ 0.85`: address and numbers carry the decision |
| `numbers_unavailable` | number state **unavailable** and `N ≥ 0.85` and `A ≥ 0.65` |

**Relative gate.** A rule-acceptable candidate becomes a **match** only if its rarity
is at least **0.10 ×** the highest rarity among the same query's rule-acceptable
candidates, across S2 and S3.

**Consequences.**
- A query may get zero, one or many matches. Nothing is ever forced, so a query
  with no rule-acceptable candidate returns `[]`.
- **Number conflicts and partial overlaps are never accepted.** Among candidates
  with similar name and address, true pairs had number Jaccard 0.79 against 0.12
  for false ones; the false positives are same-name records at other house numbers.
- Exact name and address agreement is covered by `name_address`. There is no
  shortcut that bypasses the number gate.
- Ties are deterministic: equal rarities are all kept.

## Candidate prefilter (explicit stage before the matcher)

A candidate is kept if **(no number conflict and (`A ≥ 0.60` or `N ≥ 0.75`)) or
Phase 3B blocking rank ≤ 5**.

The matcher receives only these candidates, and `candidate_pairs.tsv` is exactly this
set. Every rule path requires `A ≥ 0.60` and no conflict, so the screen can never
remove a candidate the rules could accept. `load_matcher_config` refuses any
prefilter stricter than the rules, and a property test checks this on 5,000 random
evidence rows.

- **The `N ≥ 0.75` branch and the top-5 blocking ranks** add no matcher decisions.
  They preserve recall headroom for Phase 6.
- **Nothing is dropped after scoring:** the screen runs before the matcher, and
  decisions are computed only from screened rows.

### Candidate-size Pareto (development sample, frozen matcher)

| `blocking_k10` | 19.79 | 794,828 | 20 | 20 | 87.68% | 0.94298 | **0.83550** | 1,999 / 2,314 |
| `blocking_k20` | 38.62 | 1,551,510 | 40 | 40 | 91.41% | 0.96167 | **0.84563** | 1,930 / 2,314 |
| `none` | 86.00 | 3,454,436 | 100 | 100 | 94.85% | 0.98057 | **0.85636** | 1,839 / 2,314 |
| `screen_a0.4_n0.6_k15` | 65.05 | 2,613,192 | 67 | 96 | 94.56% | 0.97933 | **0.85636** | 1,839 / 2,314 |
| `screen_a0.5_n0.7_k10` | 45.19 | 1,815,290 | 44 | 75 | 94.43% | 0.97880 | **0.85636** | 1,839 / 2,314 |
| `screen_a0.5_n0.7_k5` | 39.21 | 1,574,866 | 38 | 70 | 94.07% | 0.97751 | **0.85636** | 1,839 / 2,314 |
| `screen_a0.6_n0.75_k5` | 24.73 | 993,372 | 21 | 51 | 94.02% | 0.97737 | **0.85636** | 1,839 / 2,314 |
| `screen_a0.6_n0.8_k0` | 17.86 | 717,261 | 15 | 44 | 90.31% | 0.95880 | **0.85636** | 1,839 / 2,314 |

`none` is the Phase 3B set. `blocking_kN` keeps only Phase 3B ranks ≤ N; the
`screen_*` rows are relaxed-rule screens. A score-based top-M screen was also tried
in exploration. It ranked positives worse than the blocking rank (10 per source:
81.1% recall against 87.7%), so it was dropped.

**Selected: `screen_a0.6_n0.75_k5`.** It gives 24.7 candidates per Source 1 entity,
71% fewer than Phase 3B, with identical Macro F0.5. Recall falls only from 94.85%
to 94.02%. The smallest screen tried (17.9 per entity) also leaves matching
unchanged, but gives up 3.7 recall points, which would cap Phase 6.

## Development experiments (sample, development labels only)

| # | Design | Configurations | Best Macro F0.5 | Decision |
| ---: | --- | ---: | ---: | --- |
| 1 | `N ≥ a` and `A ≥ b` | 12 | 0.70704 | False positives from same-name siblings |
| 2 | + number gate; script-mismatch path; numbers-unavailable path | 54 | 0.84030 | Kept |
| 3 | Drop script path; add blocking-support and empty-address paths | 16 | 0.83330 | Rejected (lost true pairs) |
| 4 | + relative rarity gate (query scope, ρ ∈ {0.1, 0.2, 0.3, 0.5}) | 8 | 0.84682 (ρ = 0.1) | Kept |
| 5 | Threshold refinement with prefilter | 324 | 0.85408 | Several optima on grid edges |
| 6 | Edge extension (lower `N`, script and numbers-unavailable thresholds) | 81 | **0.85636** | **Frozen** |

The exploration used a vectorized equivalent of the per-entity formula. The frozen
configuration was re-scored with the repository's frozen `evaluate_sets` scorer and
reproduced 0.85636 exactly.

## Results

| Scope | Macro F0.5 | TP | FP | FN | Micro precision | Micro recall | Correct singletons | Singleton false merges | Zero / one / multiple predictions | Exact-match entities |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Development sample | 0.85636 | 103,676 | 6,320 | 35,177 | 94.25% | 74.67% | 1,839 / 2,314 | 475 | 3,116 / 6,108 / 30,946 | 16,704 |
| Validation (frozen, once) | 0.85376 | 567,270 | 35,783 | 194,322 | 94.07% | 74.48% | 9,483 / 12,360 | 2,877 | 16,499 / 34,411 / 169,500 | 90,637 |

The frozen validation Macro F0.5 is **0.85376**, against
0.85636 on the development sample. Micro precision and recall
are diagnostics only; they are not the challenge score.

**Candidate sets** (prefilter) against the Phase 3B input:

| Scope | Queries | Phase 3B cands / S1 | Screened cands / S1 | p50 | p95 | Max | Phase 3B recall | Screened recall | Phase 3B oracle | Screened oracle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Development sample | 40,170 | 86.00 | 24.73 | 21 | 51 | 100 | 94.85% | 94.02% | 0.98057 | 0.97737 |
| Validation | 220,410 | 86.08 | 24.78 | 21 | 51 | 100 | 94.87% | 94.01% | 0.98077 | 0.97747 |

**By country:**

| Scope | Macro F0.5 | TP | FP | FN | Micro precision | Micro recall | Correct singletons | Singleton false merges | Zero / one / multiple predictions | Exact-match entities |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Development India | 0.80310 | 39,477 | 5,221 | 16,055 | 88.32% | 71.09% | 488 / 862 | 374 | 938 / 2,600 / 12,445 | 5,091 |
| Development US | 0.89155 | 64,199 | 1,099 | 19,122 | 98.32% | 77.05% | 1,351 / 1,452 | 101 | 2,178 / 3,508 / 18,501 | 11,613 |
| Validation India | 0.80001 | 216,986 | 29,628 | 87,777 | 87.99% | 71.20% | 2,628 / 4,851 | 2,223 | 5,156 / 14,401 / 68,427 | 28,142 |
| Validation US | 0.88947 | 350,284 | 6,155 | 106,545 | 98.27% | 76.68% | 6,855 / 7,509 | 654 | 11,343 / 20,010 / 101,073 | 62,495 |

**By source:**

| Split | Source | Predicted | TP | FP | Precision | Positives in screened candidates |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Development | S2 | 54,452 | 50,873 | 3,579 | 93.43% | 63,343 |
| Development | S3 | 55,544 | 52,803 | 2,741 | 95.07% | 67,212 |
| Validation | S2 | 298,207 | 277,859 | 20,348 | 93.18% | 346,514 |
| Validation | S3 | 304,846 | 289,411 | 15,435 | 94.94% | 369,490 |

**By rule path** (a pair may satisfy several paths):

| Split | Path | Accepted | True | Precision |
| --- | --- | ---: | ---: | ---: |
| Development | `name_address` | 90,878 | 87,571 | 96.36% |
| Development | `numbers_unavailable` | 12,518 | 10,989 | 87.79% |
| Development | `script_mismatch` | 6,600 | 5,116 | 77.52% |
| Validation | `name_address` | 498,261 | 479,949 | 96.32% |
| Validation | `numbers_unavailable` | 67,741 | 59,046 | 87.16% |
| Validation | `script_mismatch` | 37,051 | 28,275 | 76.31% |

**Reading these results.**
- **Recall, not precision, is the main loss.** The matcher recovers 74.5% of links
  while screened candidates contain 94.0%. Micro precision is 94%.
- **India is the weak country.** Macro F0.5 is 0.800 against 0.889 for the US, and
  precision is 88% against 98%. Only 54–57% of Indian singletons are correctly left
  unmatched, against 91–93% in the US. On the development sample, Indian false
  positives come from all three paths: 2,206 `name_address`, 1,258
  `script_mismatch` and 1,034 `numbers_unavailable` on non-singletons.
- **S3 is more precise than S2** (95% against 93%), even though S3 addresses are
  reordered. The order-insensitive address measure handles that.

## First test submission (matcher-v1)

| Item | Value |
| --- | --- |
| Test Source 1 rows (both files) | 1,732,544 |
| Phase 3B candidates before screening | 142,805,587 (mean 82.43) |
| `candidate_pairs.tsv` candidates | 44,419,444: mean 25.64, p50 22, p90 46, p95 54, p99 71, max 100, 30 empty |
| `matching_results.tsv` matches | 5,030,493: mean 2.904, p50 3, p99 7, max 85; **112,113 empty** (6.47%) |
| `candidate_pairs.tsv` | 594,841,148 bytes, sha256 `bd0cd868dfaa017e047637994ef38fa107d4c28e50e63f575d7512fabcb77394` |
| `matching_results.tsv` | 87,280,825 bytes, sha256 `4cb9a2327cd6cecb6a131dba5b7756a2c4d8084cfa212dc7a6efe383fc1cdaee` |

| Country | Queries | Cands / S1 | p50 | p95 | Max | Matches / S1 | Max matches | Empty match lists |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| France | 259,452 | 27.61 | 24 | 60 | 100 | 3.203 | 85 | 3.99% |
| India | 809,986 | 30.90 | 28 | 59 | 100 | 2.943 | 38 | 5.66% |
| US | 663,106 | 18.44 | 16 | 34 | 100 | 2.738 | 12 | 8.43% |

Both files are written by the single centralized writer, in binary Source 1 ID
order. Within each row, S2 IDs come before S3 IDs, each in binary order. Every
match row is a subset of its candidate row. The outputs are git-ignored (`output/*`);
the hashes above identify this submission version.

**Official validator.**
- **The official command cannot run whole-file on this machine.** It holds every
  ID in Python sets. Checking both full files together was killed at a 6.5 GB
  memory cap. When `--candidate` is omitted it defaults to
  `output/candidate_pairs.tsv`, which caused the same kill.
- **Matching file: PASS.** The official CLI on `matching_results.tsv` with
  `--check-ids`, pointing `--candidate` at a nonexistent path, reported
  1,732,544 rows, 112,113 empty, and all IDs existing among 9,969,589 test S2/S3
  IDs.
- **Both files: PASS.** `scripts/validate_outputs.py` applies the official
  `validate_id_list_file` to 9 contiguous row shards per file against the full
  test ID universe. It also streams both files to check strictly increasing,
  unique Source 1 rows, exact coverage of all 1,732,544 test Source 1 IDs,
  aligned rows, and matching ⊆ candidate for every row. There were no errors or
  failures.

**Nothing was uploaded to the portal.**

## France

France has no training labels, so no French rule exists and no French result can be
measured. The pipeline runs unchanged on open country labels; a tiny end-to-end
fixture with `France` and an unseen label is tested.

**Observed risk on unlabeled test output.**
- French companies often embed the town name in their business name ("La
  Baule-Escoublac … SAS"), and same-town addresses share the town, the street type
  and small house numbers such as "10". Token-set similarity is inflated on both
  name and address, and the number gate then passes.
- The largest test match list (85 IDs) belongs to such a French entity, and 2,510
  test entities received more than 10 matches.
- France averages 3.20 matches per entity with 3.99% empty lists, against 2.74 and
  8.43% for the US. That suggests over-merging, likely including false singleton
  merges.

The frozen rules were **not** changed on the basis of unlabeled test inspection.

## Runtime, memory, disk

| Run | Queries | Pairs through features | Runtime | Peak RSS (includes mapped files) |
| --- | ---: | ---: | ---: | ---: |
| Development sample (all Pareto points) | 40,170 | 3,454,436 | 156.7 s | 4,778 MiB |
| Validation confirmation | 220,410 | 18,972,356 | 593.2 s | 5,084 MiB |
| Test (build test stores, indexes and offsets, then write files) | 1,732,544 | 142,805,587 | 4990.0 s | 6,273 MiB |
| `validate_outputs.py` | — | — | 60 s | under 6.5 GB cap |

- **New derived data (ignored):** test stores 1.8 GiB, test indexes 3.4 GiB and test
  text offsets 90 MiB, all in separate `*_test` directories so the training
  artifacts are untouched.
- **Output files:** 651 MiB.
- **Free disk** after everything: 17.83 GiB, above the 10 GiB reserve.

## Known failure modes

1. **Low recall inside the candidates.** About 19 points of recall are lost to hard
   thresholds, mostly weak names with typos or rebrandings, partial or dropped
   numbers, and empty target addresses.
2. **Indian false merges and singletons.** Indian precision is 88%, and 43–46% of
   Indian singletons are falsely merged. There is no single responsible path.
3. **French town-name inflation** (unmeasurable). See France above.
4. **No calibration.** Coarse thresholds and the relative gate cannot trade
   evidence smoothly; for example, strong rarity cannot compensate for a slightly
   low name similarity.

## Recommendation for Phase 6

Supervised modeling is **likely to improve materially**.
- **Headroom.** The screened oracle ceiling is 0.977 against the matcher's 0.854,
  a gap of 0.12 Macro F0.5 on identical candidates.
- **Evidence.** The rules use only 8 of the 42 features.
- **Where it matters most.** The largest gaps are recall and India, where a model
  can combine many weaker signals: support families, channel indicators, char
  similarities, numbers and missingness.

Suggested next steps:
1. **Train on screened development candidates.** Train a compact tabular model on
   the screened development pairs (about 24.7 per query), streamed or
   negative-sampled, never materialized in full. Tune one decision threshold per
   query context for Macro F0.5 on development; validate once.
2. **Keep the explicit prefilter.** Keep it as the candidate set, or re-derive it
   from the model's scores before the final decision stage.
3. **Add locality-aware, IDF-weighted token features.** Tokens shared between a
   business name and its own address (town names), and frequent tokens, should
   count less. This targets the French over-merging without French labels.
4. **Keep per-query relative features** (best score, margin) as model inputs.
