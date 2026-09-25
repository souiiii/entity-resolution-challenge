# Phase 3B — Deterministic candidate ranking, pruning and recovery

This phase changes candidate generation only. There are no matching predictions,
learned models, fuzzy comparisons, approximate retrieval, external data, or test
candidate exports. `blocking_v1.json`, normalization, the official writer and the
frozen evaluation protocol remain unchanged.

Every experiment uses development queries against the **complete** training
S2 (5,034,616) and S3 (5,285,603) corpora. Validation labels are discarded by the
existing development-only truth loader. The split fingerprint is
`1dd67d77fee74aa0b64477e10f09377ac22854f5eb13780f3aacc4baf94e3645`.
The evaluator additionally verifies the ground-truth input hash and full-run
query count against frozen metadata.

## Oracle diagnostic

Every operating point reports an **oracle blocking Macro F0.5 ceiling**, not a
model score. The oracle predicts exactly `truth ∩ candidates`, so FP = 0 and each
nonempty-truth entity scores `5 TP / (5 TP + FN)`. Empty truth scores 1 regardless
of how many distractor candidates the blocker emits: the oracle rejects them all.

The implementation groups `(truth count, recovered count)` pairs and calls the
existing frozen `entity_f05` helper for each group. It then takes the count-weighted
mean over all queries. This preserves the official empty-set semantics without
copying the metric formula into a second scorer. Candidate recall is a different,
link-weighted quantity. Reported oracle deltas use the unpruned Phase 3A union.

## Ranking and candidate pipeline

Each expanded, eligible blocking key contributes its full target bucket frequency.
The deduplicated candidate union retains its channel bitmask, smallest shared-key
frequency, and sum of inverse bucket frequencies. Repeated occurrences of a query
key and repeated target postings are removed before this accumulation. Distinct
keys within the same channel may contribute separately; canonical/folded exact-name
channels also each contribute their own rarity evidence.

All four trial rankings are lexicographic. Exact address support comes first, then
any canonical/folded/folded-compact exact-name support. Remaining priorities are:

| Strategy | Priorities after exact evidence |
| --- | --- |
| `channels` | Number of supporting channels descending; smallest frequency ascending |
| `families` | Supporting channel families descending; smallest frequency ascending |
| `rarity` | Smallest frequency ascending; supporting families descending |
| `aggregate` | Sum of inverse frequencies descending; supporting families descending; smallest frequency ascending |

Families collapse the three correlated exact-name views into one family; other
families are identified by channel kind and parts. This is blocking evidence,
not a pairwise text feature or a fitted score. No source-specific weights are used.
Every final tie uses target entity ID in binary order. Output remains in binary
ID order after pruning, independently within S2 and S3.

`SourceBlocker.candidates()` exposes the union for diagnostics;
`SourceBlocker.configured_candidates()` applies the selected configuration.
`prune_candidates()` is the reusable per-source top-K operation. The sample export
uses the existing centralized official-format writer and the selected candidate
set. No competing writer or `matching_results.tsv` was created.

Evaluation expands and ranks one source and one 5,000-query batch at a time. Each
truth link's rank is recorded once, so every K is evaluated in the same pass.
Per-query counts and link recovery arrays scale with queries/truth links, not with
the full candidate-pair count. No full development candidate TSV is materialized.
The Phase 3A mask is extracted from the first nine channels of this same union;
there is no second Phase 3A evaluation run.

## Recovery investigation and disk preflight

Phase 3A's most common missed-link evidence was an address word and/or a name token
whose individual bucket exceeded a cap. A name token × address word conjunction
can make that evidence specific without increasing any existing single-token cap.
It is constrained to four rarest eligible keys per query, with a small conjunction
bucket cap. Folded exact name × address number was also trialled as a compact way
to rescue capped exact names. Only these two new indexes were built.

Bounds were computed from the existing stores **before** building. At that point
25.41 GiB was free. Persistent estimates below conservatively assume every
generated posting survives; the build bound is twice that estimate, including
partition files and finished postings.

| Design | S2 generated postings | S3 generated postings | Combined persistent upper bound GiB | Largest single-source build bound GiB |
| --- | ---: | ---: | ---: | ---: |
| Folded name × number | 6,515,991 | 6,872,626 | 0.150 | 0.154 |
| Name-token pair × number | 35,641,698 | 39,631,360 | 0.841 | 0.886 |
| Three name tokens | 22,043,878 | 27,453,550 | 0.553 | 0.614 |
| Name token × address word | 119,278,205 | 122,882,885 | 2.706 | 2.747 |

The last design was investigated because of the missed-link evidence, not merely
because its Cartesian product was feasible. `index_keep=50` discards postings in
larger buckets while retaining their exact frequencies. Its actual combined index
size is about 1.29 GiB. The naive unrestricted version was not built. Three-way
keys were estimated but deferred: the two-part recovery channel provided a measured
recall gain within budget, without adding generalized three-way machinery.

All old index IDs and their contents are reused. Exact fields were allowed as
parts of the existing two-part conjunction API, a small general extension needed
for the folded-name trial. There is no normalization change. Country remains part
of every key, including for France and arbitrary unseen labels.

## Reproduction

Run in the project environment (`.venv/bin/python` on this machine):

```bash
python scripts/estimate_blocking.py --config experiments/configs/blocking_v2_recovery_sample.json
python scripts/evaluate_blocking.py --sample --config experiments/configs/blocking_v2_ranking_sample.json
python scripts/evaluate_blocking.py --sample --config experiments/configs/blocking_v2_aggregate_sample.json
python scripts/evaluate_blocking.py --sample --config experiments/configs/blocking_v2_recovery_sample.json
python scripts/evaluate_blocking.py --sample --config experiments/configs/blocking_v2_cap10_sample.json
python scripts/evaluate_blocking.py --sample --config experiments/configs/blocking_v2_cap20_sample.json
python -m pytest -q
python scripts/evaluate_blocking.py --full --config experiments/configs/blocking_v2.json
```

Each config is committed. Reports and logs are ignored under `data/interim/blocking/`.
The sample is the existing label-independent 2% hash sample, salt
`blocking-sample-v1`: 40,170 queries, 67,262 S2 links and 71,591 S3 links.
Every sample report contains source/country recall, entity recovery, candidate
percentiles and oracle ceilings for every evaluated ranking/K pair. Sample
experiments preceded configuration selection. Exactly one final full-development
Phase 3B run was then performed; its complete curves were computed together.

## Sample experiments and selection

| Experiment | Ranking (K=50) | Recall | Pairs | Mean | Oracle ceiling | Reduction vs Phase 3A |
| --- | --- | --- | --- | --- | --- | --- |
| ranking | channels | 92.8226% | 3,364,703 | 83.762 | 0.97051427 | 40.5984% |
| ranking | families | 92.8226% | 3,364,703 | 83.762 | 0.97051427 | 40.5984% |
| ranking | rarity | 92.7837% | 3,364,703 | 83.762 | 0.97037093 | 40.5984% |
| aggregate | aggregate | 92.9256% | 3,364,703 | 83.762 | 0.97096443 | 40.5984% |
| recovery | families | 94.7455% | 3,454,479 | 85.996 | 0.98017531 | 39.0135% |
| recovery | aggregate | 94.8557% | 3,454,479 | 85.996 | 0.98057434 | 39.0135% |
| cap10 | aggregate | 94.8550% | 3,454,436 | 85.995 | 0.98057019 | 39.0143% |
| cap20 | aggregate | 94.9112% | 3,495,447 | 87.016 | 0.98088429 | 38.2902% |

All oracle columns in this report mean **oracle blocking Macro F0.5 ceiling**, never a model score.

Baseline-only ranking comparison (same candidate volume at each K):

| Ranking | K=10 | K=20 | K=30 | K=50 | K=75 | K=100 | K=150 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| channels | 0.92763645 | 0.94977277 | 0.95747876 | 0.97051427 | 0.97332091 | 0.97399905 | 0.97449346 |
| families | 0.92785509 | 0.94987061 | 0.95752254 | 0.97051427 | 0.97332091 | 0.97399905 | 0.97449346 |
| rarity | 0.92732125 | 0.94955380 | 0.95729670 | 0.97037093 | 0.97331450 | 0.97399767 | 0.97449053 |
| aggregate | 0.93215317 | 0.95177310 | 0.95849884 | 0.97096443 | 0.97352665 | 0.97408911 | 0.97451731 |

Recovery contribution before pruning, in configuration order after the original nine channels:

| Experiment | Channel | Standalone pairs | Standalone true | Added pairs | Added true | Recall gain | True / 1,000 added |
| --- | --- | --- | --- | --- | --- | --- | --- |
| recovery | folded_name_x_number | 30,371 | 27,634 | 113 | 3 | 0.0022 pp | 26.549 |
| recovery | name_x_address_word | 448,714 | 100,775 | 298,372 | 2,478 | 1.7846 pp | 8.305 |
| cap10 | name_x_address_word | 448,714 | 100,775 | 298,374 | 2,480 | 1.7861 pp | 8.312 |
| cap20 | name_x_address_word | 648,972 | 102,598 | 493,387 | 2,659 | 1.9150 pp | 5.389 |

The folded-name conjunction is rejected from the final configuration: only three
incremental sample links for an additional ~150 MiB index. Without it the word
conjunction independently adds 2,480 links (two of those three were redundant with
it). Its ranking evidence changed a few ties; the direct cap-10 rerun measures the
exact final channel set rather than assuming removal has no effect.

The cap-20 alternative remains on the Pareto frontier but has diminishing returns.
At K=50 it adds 41,011 pairs (1.021 per query), 78 true links, and 0.00031410 oracle
score over cap 10. The smaller cap is selected. No existing single-token cap was
increased. No generic token × word index without bucket limits was constructed.

The selected ranking is `aggregate`, with **K=50 separately for S2 and S3**.
K=40 nearly preserves the old union's oracle ceiling, but sacrifices 1,535 recovered
sample links versus K=50; K=75 costs 1,089,893 additional pairs for 693 more links.
K=50 is a measured middle point, not a universal optimum or a model threshold.
It reduces sample volume by 39.0143%, increases recall by 1.0493 percentage points,
and increases oracle ceiling by 0.00595395 over Phase 3A. Relative to the expanded
unpruned union, pruning loses 0.7368 recall points and 0.00288942 oracle score.

The exact selected sample curve follows. Entity recovery percentages use the
nonempty-truth queries as denominator; oracle means include all queries.

| Point | S2 recall | S3 recall | Overall recall | India recall | US recall | Oracle blocking Macro F0.5 ceiling | Oracle Δ vs Phase 3A |
| --- | --- | --- | --- | --- | --- | --- | --- |
| phase3a_union | 93.8479% | 93.7660% | 93.8057% | 92.1973% | 94.8776% | 0.97461624 | +0.00000000 |
| unpruned | 95.5993% | 95.5846% | 95.5917% | 93.1211% | 97.2384% | 0.98345961 | +0.00884337 |
| aggregate@10 | 89.1960% | 86.2511% | 87.6776% | 84.0957% | 90.0649% | 0.94298316 | -0.03163307 |
| aggregate@20 | 91.7472% | 91.0952% | 91.4111% | 90.2273% | 92.2000% | 0.96167286 | -0.01294338 |
| aggregate@30 | 92.8845% | 92.4530% | 92.6620% | 91.3942% | 93.5070% | 0.96836981 | -0.00624642 |
| aggregate@40 | 93.9059% | 93.6025% | 93.7495% | 91.9146% | 94.9725% | 0.97465515 | +0.00003891 |
| aggregate@50 | 94.8901% | 94.8220% | 94.8550% | 92.2909% | 96.5639% | 0.98057019 | +0.00595395 |
| aggregate@75 | 95.3867% | 95.3234% | 95.3541% | 92.7321% | 97.1016% | 0.98253326 | +0.00791702 |
| aggregate@100 | 95.5012% | 95.4631% | 95.4816% | 92.9554% | 97.1652% | 0.98298380 | +0.00836756 |
| aggregate@150 | 95.5785% | 95.5609% | 95.5694% | 93.0887% | 97.2228% | 0.98335866 | +0.00874242 |

| Point | Pairs | Mean / S1 | p50 | p90 | p95 | p99 | Max | Zero candidates | Reduction vs Phase 3A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase3a_union | 5,664,334 | 141.009 | 128 | 255 | 298 | 391 | 774 | 3 | 0.0000% |
| unpruned | 5,962,708 | 148.437 | 136 | 263 | 307 | 401 | 803 | 0 | -5.2676% |
| aggregate@10 | 794,828 | 19.787 | 20 | 20 | 20 | 20 | 20 | 0 | 85.9678% |
| aggregate@20 | 1,551,510 | 38.624 | 40 | 40 | 40 | 40 | 40 | 0 | 72.6091% |
| aggregate@30 | 2,251,529 | 56.050 | 60 | 60 | 60 | 60 | 60 | 0 | 60.2508% |
| aggregate@40 | 2,887,604 | 71.885 | 80 | 80 | 80 | 80 | 80 | 0 | 49.0213% |
| aggregate@50 | 3,454,436 | 85.995 | 100 | 100 | 100 | 100 | 100 | 0 | 39.0143% |
| aggregate@75 | 4,544,329 | 113.127 | 127 | 150 | 150 | 150 | 150 | 0 | 19.7729% |
| aggregate@100 | 5,219,518 | 129.936 | 135 | 200 | 200 | 200 | 200 | 0 | 7.8529% |
| aggregate@150 | 5,789,599 | 144.127 | 136 | 256 | 285 | 300 | 300 | 0 | -2.2115% |

| Point | All true links recovered | At least one recovered | Zero recovered |
| --- | --- | --- | --- |
| phase3a_union | 31,702 (83.7437%) | 37,461 (98.9566%) | 395 (1.0434%) |
| unpruned | 33,105 (87.4498%) | 37,637 (99.4215%) | 219 (0.5785%) |
| aggregate@10 | 27,687 (73.1377%) | 36,823 (97.2712%) | 1,033 (2.7288%) |
| aggregate@20 | 30,156 (79.6598%) | 37,191 (98.2433%) | 665 (1.7567%) |
| aggregate@30 | 31,000 (81.8893%) | 37,337 (98.6290%) | 519 (1.3710%) |
| aggregate@40 | 31,674 (83.6697%) | 37,483 (99.0147%) | 373 (0.9853%) |
| aggregate@50 | 32,380 (85.5347%) | 37,601 (99.3264%) | 255 (0.6736%) |
| aggregate@75 | 32,854 (86.7868%) | 37,626 (99.3924%) | 230 (0.6076%) |
| aggregate@100 | 32,984 (87.1302%) | 37,630 (99.4030%) | 226 (0.5970%) |
| aggregate@150 | 33,081 (87.3864%) | 37,636 (99.4189%) | 220 (0.5811%) |

## Frozen final configuration

`experiments/configs/blocking_v2.json`: version `blocking-v2`, batch size 5,000,
index-build chunk size 50,000, six store workers, 10 GiB free-disk reserve.
Ranking `aggregate`; selected K=50; measured K=[10,20,30,40,50,75,100,150].
The first nine channels define the unchanged Phase 3A baseline.
All channels apply to both sources. The sample salt/fraction remain unchanged.

| Channel | Parts | Kind | Frequency cap | Max keys | Index keep |
| --- | --- | --- | --- | --- | --- |
| address_exact | address_canonical | exact | 50 | all | 200 |
| name_exact | name_canonical | exact | 50 | all | 200 |
| name_folded | name_folded | exact | 50 | all | 200 |
| name_folded_compact | name_folded_compact | exact | 50 | all | 200 |
| name_rare_token | name | token | 30 | 1 | 200 |
| name_token_pair | name × name (unordered) | combo | 50 | 8 | 200 |
| name_token_x_address_number | name × address_number | combo | 50 | 8 | 200 |
| address_number_x_word | address_number × address_word | combo | 50 | 8 | 200 |
| address_rare_word | address_word | token | 20 | 2 | 200 |
| name_x_address_word | name × address_word | combo | 10 | 4 | 50 |


## Single final full-development evaluation

The frozen configuration was evaluated once on all 1,986,411 development queries,
with 3,325,308 S2 and 3,551,465 S3 links. Extracting the first nine channel bits
reproduced **every saved Phase 3A recall, candidate-distribution and entity-recovery
field exactly**. Those saved metrics are the baseline; the previously unavailable
baseline oracle ceiling is computed from recovery counts within this same pass.
No parameter was changed after inspecting the full-development results.

| Metric | Phase 3A union | Phase 3B selected | Change |
| --- | --- | --- | --- |
| Candidate pairs | 280,185,604 | 170,895,201 | 39.0064% fewer |
| Mean candidates / S1 | 141.051 | 86.032 | -55.019 |
| Link recall | 93.7969% | 94.8223% | +1.0254 pp |
| Oracle blocking Macro F0.5 ceiling | 0.97481258 | 0.98059431 | +0.00578173 |
| All true links recovered | 83.7985% | 85.6372% | +34,485 entities |
| Zero true links recovered | 19,009 | 12,343 | -6,666 entities |

Relative to the **expanded unpruned union**, K=50 loses 52,989 true links, 0.7706 recall points and 0.00299499 oracle score. These pruning losses are real despite the net improvement over Phase 3A.

| Point | S2 recall | S3 recall | Overall recall | India recall | US recall | Oracle blocking Macro F0.5 ceiling | Oracle Δ vs Phase 3A |
| --- | --- | --- | --- | --- | --- | --- | --- |
| phase3a_union | 93.8070% | 93.7875% | 93.7969% | 92.0916% | 94.9368% | 0.97481258 | +0.00000000 |
| unpruned | 95.5940% | 95.5918% | 95.5929% | 93.0730% | 97.2773% | 0.98358930 | +0.00877673 |
| aggregate@10 | 88.9677% | 86.0302% | 87.4506% | 83.7649% | 89.9143% | 0.94148641 | -0.03332617 |
| aggregate@20 | 91.5807% | 91.0154% | 91.2887% | 90.1212% | 92.0692% | 0.96110014 | -0.01371244 |
| aggregate@30 | 92.7769% | 92.3958% | 92.5801% | 91.2936% | 93.4400% | 0.96811919 | -0.00669339 |
| aggregate@40 | 93.8063% | 93.5407% | 93.6691% | 91.8705% | 94.8714% | 0.97416003 | -0.00065255 |
| aggregate@50 | 94.8515% | 94.7950% | 94.8223% | 92.2261% | 96.5577% | 0.98059431 | +0.00578173 |
| aggregate@75 | 95.3494% | 95.3483% | 95.3488% | 92.6832% | 97.1306% | 0.98264869 | +0.00783611 |
| aggregate@100 | 95.4766% | 95.4857% | 95.4813% | 92.8884% | 97.2145% | 0.98315315 | +0.00834058 |
| aggregate@150 | 95.5678% | 95.5719% | 95.5700% | 93.0336% | 97.2653% | 0.98349762 | +0.00868504 |

| Point | Pairs | Mean / S1 | p50 | p90 | p95 | p99 | Max | Zero candidates | Reduction vs Phase 3A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase3a_union | 280,185,604 | 141.051 | 129 | 254 | 298 | 388 | 885 | 251 | 0.0000% |
| unpruned | 295,001,986 | 148.510 | 136 | 263 | 307 | 399 | 891 | 23 | -5.2881% |
| aggregate@10 | 39,309,389 | 19.789 | 20 | 20 | 20 | 20 | 20 | 23 | 85.9702% |
| aggregate@20 | 76,728,142 | 38.627 | 40 | 40 | 40 | 40 | 40 | 23 | 72.6152% |
| aggregate@30 | 111,358,892 | 56.060 | 60 | 60 | 60 | 60 | 60 | 23 | 60.2553% |
| aggregate@40 | 142,835,095 | 71.906 | 80 | 80 | 80 | 80 | 80 | 23 | 49.0213% |
| aggregate@50 | 170,895,201 | 86.032 | 100 | 100 | 100 | 100 | 100 | 23 | 39.0064% |
| aggregate@75 | 224,877,729 | 113.208 | 128 | 150 | 150 | 150 | 150 | 23 | 19.7397% |
| aggregate@100 | 258,292,199 | 130.030 | 135 | 200 | 200 | 200 | 200 | 23 | 7.8139% |
| aggregate@150 | 286,535,055 | 144.248 | 136 | 256 | 286 | 300 | 300 | 23 | -2.2662% |

| Point | All true links recovered | At least one recovered | Zero recovered |
| --- | --- | --- | --- |
| phase3a_union | 1,571,661 (83.7985%) | 1,856,515 (98.9865%) | 19,009 (1.0135%) |
| unpruned | 1,642,121 (87.5553%) | 1,865,136 (99.4461%) | 10,388 (0.5539%) |
| aggregate@10 | 1,366,171 (72.8421%) | 1,822,023 (97.1474%) | 53,501 (2.8526%) |
| aggregate@20 | 1,491,199 (79.5084%) | 1,842,051 (98.2153%) | 33,473 (1.7847%) |
| aggregate@30 | 1,535,060 (81.8470%) | 1,849,667 (98.6213%) | 25,857 (1.3787%) |
| aggregate@40 | 1,570,226 (83.7220%) | 1,856,285 (98.9742%) | 19,239 (1.0258%) |
| aggregate@50 | 1,606,146 (85.6372%) | 1,863,181 (99.3419%) | 12,343 (0.6581%) |
| aggregate@75 | 1,629,751 (86.8958%) | 1,864,528 (99.4137%) | 10,996 (0.5863%) |
| aggregate@100 | 1,636,440 (87.2524%) | 1,864,854 (99.4311%) | 10,670 (0.5689%) |
| aggregate@150 | 1,640,929 (87.4918%) | 1,865,076 (99.4429%) | 10,448 (0.5571%) |

### Selected source/country breakdown

| Country | Source | Recovered links | True links | Recall |
| --- | --- | --- | --- | --- |
| India | overall | 2,540,904 | 2,755,080 | 92.2261% |
| India | source2 | 1,231,599 | 1,332,937 | 92.3974% |
| India | source3 | 1,309,305 | 1,422,143 | 92.0656% |
| US | overall | 3,979,812 | 4,121,693 | 96.5577% |
| US | source2 | 1,922,506 | 1,992,371 | 96.4934% |
| US | source3 | 2,057,306 | 2,129,322 | 96.6179% |

| Scope | Queries | Pairs | Mean | p50 | p90 | p95 | p99 | Max | Zero candidates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| source2 | 1986411 | 86,167,692 | 43.379 | 50 | 50 | 50 | 50 | 50 | 155 |
| source3 | 1986411 | 84,727,509 | 42.654 | 50 | 50 | 50 | 50 | 50 | 135 |
| India | 795204 | 68,236,161 | 85.810 | 100 | 100 | 100 | 100 | 100 | 14 |
| US | 1191207 | 102,659,040 | 86.181 | 100 | 100 | 100 | 100 | 100 | 9 |

All 110,887 true singletons score 1 under the oracle. They receive 9,646,629 candidate pairs in total (median 100, p99 100, 7 with none). These are candidates, not false-positive matches; the future matcher must reject them.


| Country | Source | Phase 3A recall | Selected recall | Change |
| --- | --- | --- | --- | --- |
| India | source2 | 92.5502% | 92.3974% | -0.1528 pp |
| India | source3 | 91.6618% | 92.0656% | +0.4038 pp |
| India | overall | 92.0916% | 92.2261% | +0.1345 pp |
| US | source2 | 94.6478% | 96.4934% | +1.8456 pp |
| US | source3 | 95.2072% | 96.6179% | +1.4107 pp |
| US | overall | 94.9368% | 96.5577% | +1.6209 pp |

| Country | Phase 3A all-links recovery | Selected all-links recovery | Change |
| --- | --- | --- | --- |
| India | 610,417 (81.3126%) | 603,073 (80.3343%) | -0.9783 pp |
| US | 961,244 (85.4576%) | 1,003,073 (89.1763%) | +3.7187 pp |

**The aggregate improvement is uneven.** India S2 loses 2,037 links, and India
has 7,344 fewer entities with every link recovered. Its overall link recall rises
slightly and its zero-recovery count improves, but those do not erase the loss in
complete entity recovery. The configuration remains the one chosen on the sample;
these full-development results were not used for further tuning.

### New-channel contribution and pruning accounting

| Source | Standalone candidates | Standalone true links | Standalone recall | Incremental candidates | Incremental true links | Recall gain | True / 1,000 added |
| --- | --- | --- | --- | --- | --- | --- | --- |
| source2 | 10,845,123 | 2,449,037 | 73.6484% | 7,164,466 | 59,423 | 1.7870 pp | 8.294 |
| source3 | 11,397,590 | 2,524,097 | 71.0720% | 7,651,916 | 64,082 | 1.8044 pp | 8.375 |

| Source | New true links before pruning | New true links retained | Phase 3A links lost | Net recovered-link change |
| --- | --- | --- | --- | --- |
| source2 | 59,423 | 57,304 | 22,570 | +34,734 |
| source3 | 64,082 | 62,196 | 26,414 | +35,782 |

Incremental counts are after the unchanged nine channels. Standalone evidence
can overlap them. The two accounting identities above were checked exactly for
each source, as were candidate/source totals and monotonicity across K.

## Resources and verification

| Item | Measured value |
| --- | --- |
| Full-run wall time | 498.0 s |
| S2 / S3 pass time | 192.4 / 207.9 s |
| Peak process RSS (includes mapped file pages) | 5,257 MiB |
| Active stores + indexes | 5.173 GiB |
| All retained blocking stores + indexes | 5.321 GiB (includes rejected folded-name trial indexes) |
| Additional retained persistent indexes | 1.441 GiB |
| Free disk before / after full run | 23.96 / 23.94 GiB |
| Configured free-disk reserve | 10 GiB |
| Tests | 187 passed; all original 169 retained |

The measured 5,257 MiB peak RSS is higher than Phase 3A's reported ~3.2 GiB.
This run was not placed in Phase 3A's 2.8 GB systemd memory scope; RSS includes
reclaimable mapped pages, so it is not a measurement of anonymous memory alone.
The extra index and per-K query/link arrays raise the working set. Full runtime
is 498 s versus the saved Phase 3A 325 s; candidate reduction does not imply that
blocking itself became faster.

The full run reused all indexes; its runtime includes input/manifest verification,
truth loading, both source passes, all curves and diagnostics. Builds use disk
partitions and memory maps, with 50,000-row generation chunks. No failed `.building`
or `.partial` artifacts remain. The existing multiprocessing test needed an
unsandboxed local Unix socket; the complete suite passed with that permission.

Sample resource measurements (runtime includes the one-time build only in the
recovery trial; cache state varies, so these are not controlled speed comparisons):

| Experiment | Wall seconds | Peak RSS MiB | Index-build seconds S2 / S3 |
| --- | --- | --- | --- |
| ranking | 92.8 | 2530 | 0.0 / 0.0 |
| aggregate | 42.5 | 2571 | 0.0 / 0.0 |
| recovery | 168.4 | 4347 | 58.0 / 40.2 |
| cap10 | 68.9 | 3711 | 0.0 / 0.0 |
| cap20 | 41.5 | 3713 | 0.0 / 0.0 |

Tests cover deterministic ranking and ID ties, exact evidence priority, minimum
and aggregate rarity, independent per-source top-K, deduplication, empty candidate
sets, configured pipeline output, oracle empty/partial/full recovery, open-country
isolation, exact/token conjunctions, caps, key limits, target-order invariance,
curve accounting and invalid ranking configuration. A separate 20,000-entity
synthetic oracle audit exactly matched the frozen scorer averaged with `math.fsum`.

## Remaining failures and Phase 3C recommendation

| Union miss diagnostic | S2 | S3 |
| --- | --- | --- |
| Missed links before top-K | 146,514 | 156,554 |
| No shared deterministic key | 115 | 1,559 |
| Shared key suppressed by caps/key limits | 146,399 | 154,995 |
| Empty target address | 26,986 | 34,605 |

After K=50 there are 356,057 missing links: 303,068 absent from the expanded union and 52,989 removed by ranking. Union diagnostic counts deliberately exclude the pruning losses: a top-K exclusion must not be mislabeled a key-limit failure.

Common name/locality evidence still exceeds the conjunction cap. Empty or heavily
changed target addresses leave weak name evidence; Indic-script name differences
remain outside this deterministic design. India remains the weaker country.
Exact-name evidence can still be ambiguous for common business names. Candidate
rank is not a calibrated probability and should not be used as a match decision.

**Defer broad Phase 3C approximate retrieval.** Only 1,674 links (0.0243% of development links) have no shared deterministic key. Most remaining misses involve common evidence or pruning. A small, targeted
approximate-retrieval pilot may be justified later for unresolved queries, but
its value must be established on this same development protocol and must repay
its candidate cost. These results do not prove that a particular confidence or
low-candidate gate would find the hard misses; no such gate was tested here.
Proceed to pairwise feature/matcher work with the measured candidate ceiling,
then use its failure analysis to decide whether that targeted pilot is needed.
