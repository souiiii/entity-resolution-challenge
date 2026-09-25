# Phase 3A — Measured Deterministic Blocking Baseline

This phase generates candidates only. It adds no matching decisions, pair scoring,
fuzzy or n-gram retrieval, embeddings, classifiers, thresholds, or validation/test
predictions. The frozen split, its fingerprint
(`1dd67d77fee74aa0b64477e10f09377ac22854f5eb13780f3aacc4baf94e3645`), the
evaluator and the Phase 2 normalization (`canonical-v1`) are unchanged.

**Validation labels were not used.** Every measurement and design decision here
used development Source 1 labels only:
- The query set is the frozen development Source 1 manifest.
- Truth rows of validation entities are discarded while the truth file is streamed.

As the protocol requires, development queries search the **complete** training
Source 2 + Source 3 corpus: 5,034,616 + 5,285,603 records, including
validation-owned records. Key frequencies are computed from that unlabeled target
corpus only.

## Reproduction

```bash
python scripts/evaluate_blocking.py --sample                  # hash-selected development sample
python scripts/evaluate_blocking.py --sample --write-sample-candidates   # plus a 2,000-row debug file
python scripts/evaluate_blocking.py --full                    # all development Source 1 entities
python scripts/evaluate_blocking.py --clean                   # delete derived stores and indexes
```

Outputs go to ignored `data/interim/blocking/`: a compact JSON report per mode, the
key stores (`store/`) and the cached indexes (`index/`). Terminal output is a short
summary.

## Design

Code lives in `src/blocking/`; configuration in `experiments/configs/blocking_v1.json`.

1. **Key store** (`store.py`, `keys.py`). Each source file is streamed once through
   the Phase 2 normalizer on six worker processes. Every record gets:
   - four exact keys: canonical name, folded name, folded compact name and
     canonical address;
   - its distinct folded-name token keys;
   - its ordered canonical-address token keys, each tagged as a word or a number.

   Keys are 64-bit BLAKE2b hashes of `(kind, raw country label, value)`. Country is
   part of every key, so candidates can only share a country, and an unseen label
   such as `France` forms its own blocks with no code change. Address numbers are
   hashed in integer-equivalent form (`015115` ≡ `15115`); the canonical text
   itself is untouched. Stores hold IDs and hashes only, never normalized text.
   They are rebuilt only when the raw input hash, normalization version or store
   version changes.
2. **Posting indexes** (`index.py`). For each target source and channel, an index
   holds `(key, record)` postings sorted by key.
   - Build is external: postings are split into 16 partitions by their top 4 hash
     bits, and each partition is sorted and deduplicated separately.
   - Buckets larger than `index_keep` (200) are stored only as `(key, count)`,
     because no channel may expand them. Caps are still enforced and reported
     exactly.
   - Indexes are memory-mapped and cached. Query-time limits do not change an
     index, so caps can be re-tuned without a rebuild.
3. **Channels** (`channels.py`). Each channel turns a query batch into keys and
   looks them up in bulk with numpy. It skips keys whose target frequency exceeds
   `max_frequency`, and optionally keeps only the query's `max_keys` rarest
   eligible keys (ties broken by key value).
4. **Union** (`candidates.py`). Candidates from all channels are deduplicated per
   (query, record), with an OR-ed per-candidate channel bitmask. Order is
   deterministic: target ID in binary order within each source, and all S2
   candidates before S3. The bitmask gives each channel's standalone contribution
   and its incremental contribution (credited to the first channel, in config
   order, that produced the candidate) in the same pass.
5. **Evaluation** (`evaluate.py`). Query batches are streamed per target source, so
   only one source's indexes are touched at a time. Metrics accumulate in fixed
   per-query arrays, and no development candidate pair is written to disk. After
   each source, every missed link is diagnosed: does the query share any key with
   its target, and was that key excluded by a frequency cap or a key limit?
6. **Official writer** (`src/submission/create_submission.py`). A single streaming
   writer handles both official files. It writes exact headers and
   `S1<TAB>id,id,...` rows, rejects duplicate rows, repeated targets, and
   non-`S2-`/`S3-` targets, and writes through a temporary file. Its output passes
   the official validator's parser (tested). No `candidate_pairs.tsv` or
   `matching_results.tsv` for test was produced.

### Channels and configuration

The channels run in this order, per target source:

| # | Channel | Key | Max frequency | Max keys per query |
| ---: | --- | --- | ---: | ---: |
| 1 | `address_exact` (A) | canonical address | 50 | all |
| 2 | `name_exact` (B) | canonical name | 50 | all |
| 3 | `name_folded` (C) | Latin-folded name | 50 | all |
| 4 | `name_folded_compact` (D) | folded name with spaces removed | 50 | all |
| 5 | `name_rare_token` (E) | single folded-name token | 30 | 1 (rarest) |
| 6 | `name_token_pair` (E) | unordered pair of folded-name tokens | 50 | 8 (rarest) |
| 7 | `name_token_x_address_number` (E×F) | name token × integer-equivalent address number | 50 | 8 |
| 8 | `address_number_x_word` (F) | address number × non-numeric address token | 50 | 8 |
| 9 | `address_rare_word` (F) | single non-numeric address token | 20 | 2 (rarest) |

```json
{
  "version": "blocking-v1",
  "batch_size": 20000,
  "workers": 6,
  "min_free_disk_gib": 3.0,
  "index_keep": 200,
  "sample": {"salt": "blocking-sample-v1", "fraction": 0.02},
  "channels": [
    {"name": "address_exact", "kind": "exact", "parts": ["address_canonical"], "max_frequency": 50},
    {"name": "name_exact", "kind": "exact", "parts": ["name_canonical"], "max_frequency": 50},
    {"name": "name_folded", "kind": "exact", "parts": ["name_folded"], "max_frequency": 50},
    {"name": "name_folded_compact", "kind": "exact", "parts": ["name_folded_compact"], "max_frequency": 50},
    {"name": "name_rare_token", "kind": "token", "parts": ["name"], "max_frequency": 30, "max_keys": 1},
    {"name": "name_token_pair", "kind": "combo", "parts": ["name", "name"], "symmetric": true, "max_frequency": 50, "max_keys": 8},
    {"name": "name_token_x_address_number", "kind": "combo", "parts": ["name", "address_number"], "max_frequency": 50, "max_keys": 8},
    {"name": "address_number_x_word", "kind": "combo", "parts": ["address_number", "address_word"], "max_frequency": 50, "max_keys": 8},
    {"name": "address_rare_word", "kind": "token", "parts": ["address_word"], "max_frequency": 20, "max_keys": 2}
  ]
}
```

Exploration on an earlier 39,832-query hash sample of development links drove the
choice of combination keys over single tokens. Single tokens are too common: the rarest name token shared by a
genuine link has median frequency 333 (S2) / 478 (S3), so a single-token cap of 200
reaches only about 45% of links. Pairs of keys are far more selective. With a cap of
50, the union of the three combination keys covered about 90.6% of sample links.
Name token × address word was rejected: it needs about 120M postings per source.

A separate canonical-compact name channel was not added. Equal canonical-compact
names imply equal folded-compact names, so channel 4 already covers it.

## Development sample and full run

- **Sample.** Development Source 1 IDs with `sha256("blocking-sample-v1\0" + id) <
  0.02 · 2²⁵⁶`: **40,170 queries** (2.02% of development), 67,262 S2 links and
  71,591 S3 links. Selection uses no labels. The sample was used only to debug
  counts and memory. It was run with one configuration, then rerun once to confirm
  the index-size change reproduced every metric exactly. Sample recall was 93.81%
  at a mean of 141.0 candidates, against 93.80% and 141.1 for the full run.
- **Full run.** Exactly one full development evaluation, with the committed
  configuration: **1,986,411 queries**, **3,325,308 S2** and **3,551,465 S3**
  positive links. These equal the frozen metadata; the script refuses to report
  otherwise. No parameter was changed after seeing full-development results.

## Link-level blocking recall (full development)

| Scope | S1→S2 | S1→S3 | Overall |
| --- | --- | --- | --- |
| All development | 93.81% (3,119,371 / 3,325,308) | 93.79% (3,330,829 / 3,551,465) | 93.80% (6,450,200 / 6,876,773) |
| India | 92.55% (1,233,636 / 1,332,937) | 91.66% (1,303,562 / 1,422,143) | 92.09% (2,537,198 / 2,755,080) |
| US | 94.65% (1,885,735 / 1,992,371) | 95.21% (2,027,267 / 2,129,322) | 94.94% (3,913,002 / 4,121,693) |

The matcher's link-level recall ceiling is **93.80%**. India trails the US by 2.85
points, mostly on S3 (91.66%).

## Entity-level recovery (full development)

| Measure | All | India | US |
| --- | ---: | ---: | ---: |
| Entities with ≥1 true link | 1,875,524 | 750,704 | 1,124,820 |
| All true links recovered | 1,571,661 (83.80%) | 610,417 (81.31%) | 961,244 (85.46%) |
| At least one true link recovered | 1,856,515 (98.99%) | — | — |
| Zero true links recovered | 19,009 (1.01%) | 12,824 (1.71%) | 6,185 (0.55%) |
| True singletons | 110,887 | | |

**Singletons.** All but 94 of the 110,887 true singletons receive candidates:
median 132, p99 405. The later matcher must reject every one of their candidates
to earn the singleton's 1.0; blocking cannot make that decision.

## Candidate-count distributions (full development)

| Scope | Queries | Total pairs | Mean | p50 | p90 | p95 | p99 | Max | >50 | >100 | >250 | >500 | Zero |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S2 | 1,986,411 | 142,855,174 | 71.9 | 65 | 132 | 156 | 206 | 480 | 63.73% | 23.33% | 0.22% | 0.00% | 1,893 |
| S3 | 1,986,411 | 137,330,430 | 69.1 | 62 | 130 | 154 | 204 | 471 | 60.64% | 21.78% | 0.20% | 0.00% | 1,834 |
| Combined | 1,986,411 | 280,185,604 | 141.1 | 129 | 254 | 298 | 388 | 885 | 87.05% | 64.07% | 10.61% | 0.12% | 251 |
| Combined, India | 795,204 | 121,676,840 | 153.0 | 143 | 277 | 319 | 410 | 885 | 86.82% | 67.88% | 14.61% | 0.19% | 159 |
| Combined, US | 1,191,207 | 158,508,764 | 133.1 | 121 | 237 | 278 | 367 | 773 | 87.20% | 61.52% | 7.94% | 0.08% | 92 |
| Combined, singletons | 110,887 | 16,104,857 | 145.2 | 132 | 266 | 312 | 405 | 724 | 86.68% | 64.75% | 12.49% | 0.18% | 94 |

**Volume.** In total there are 280,185,604 candidate pairs, a mean of 141 per
Source 1 entity out of 10,320,219 searchable records per query (a reduction ratio
of 99.9986%). Only 0.12% of queries exceed 500 combined candidates, and no query
exceeds 500 in a single source. There is no global per-query cap, so no candidate
was silently discarded; every limit is a documented per-channel frequency cap or
key limit.

**Largest sets.** The largest combined sets (max 885, `S1-671959997`, *City
Products Private Limited*, Secunderabad) are long Indian addresses containing many
plot, house and flat numbers, combined with generic name tokens (`private
limited`). Each combination channel can then expand 8 keys × 50 postings.

## Per-channel contributions (full development)

Incremental figures credit each candidate to the first channel, in configuration
order, that produced it. Incremental true links sum exactly to the recovered links,
and incremental candidates sum exactly to the total pairs.

| # | Channel | Source | Standalone candidates | Standalone true | Standalone recall | Incremental true | Incremental share of links | Incremental candidates | Incremental candidates / S1 | True per 1,000 added pairs |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `address_exact` | S2 | 508,182 | 415,163 | 12.48% | 415,163 | 12.48% | 508,182 | 0.26 | 817.0 |
| 1 | `address_exact` | S3 | 185,011 | 153,479 | 4.32% | 153,479 | 4.32% | 185,011 | 0.09 | 829.6 |
| 2 | `name_exact` | S2 | 5,248,792 | 679,440 | 20.43% | 594,190 | 17.87% | 5,163,542 | 2.60 | 115.1 |
| 2 | `name_exact` | S3 | 6,072,949 | 753,718 | 21.22% | 753,600 | 21.22% | 6,072,831 | 3.06 | 124.1 |
| 3 | `name_folded` | S2 | 5,485,073 | 796,635 | 23.96% | 107,303 | 3.23% | 686,856 | 0.35 | 156.2 |
| 3 | `name_folded` | S3 | 6,393,339 | 885,710 | 24.94% | 138,208 | 3.89% | 849,520 | 0.43 | 162.7 |
| 4 | `name_folded_compact` | S2 | 5,404,312 | 832,941 | 25.05% | 34,362 | 1.03% | 179,890 | 0.09 | 191.0 |
| 4 | `name_folded_compact` | S3 | 6,284,050 | 916,504 | 25.81% | 34,275 | 0.97% | 165,998 | 0.08 | 206.5 |
| 5 | `name_rare_token` | S2 | 4,699,323 | 502,355 | 15.11% | 301,285 | 9.06% | 4,445,135 | 2.24 | 67.8 |
| 5 | `name_rare_token` | S3 | 4,613,640 | 519,059 | 14.62% | 332,176 | 9.35% | 4,374,253 | 2.20 | 75.9 |
| 6 | `name_token_pair` | S2 | 39,697,340 | 2,083,858 | 62.67% | 924,701 | 27.81% | 36,384,047 | 18.32 | 25.4 |
| 6 | `name_token_pair` | S3 | 33,646,687 | 2,154,839 | 60.67% | 1,022,939 | 28.80% | 30,697,338 | 15.45 | 33.3 |
| 7 | `name_token_x_address_number` | S2 | 38,134,746 | 2,091,382 | 62.89% | 326,987 | 9.83% | 35,821,325 | 18.03 | 9.1 |
| 7 | `name_token_x_address_number` | S3 | 38,578,386 | 2,323,211 | 65.42% | 467,511 | 13.16% | 36,213,781 | 18.23 | 12.9 |
| 8 | `address_number_x_word` | S2 | 60,417,514 | 2,500,848 | 75.21% | 357,880 | 10.76% | 56,804,842 | 28.60 | 6.3 |
| 8 | `address_number_x_word` | S3 | 59,279,681 | 2,597,859 | 73.15% | 364,086 | 10.25% | 55,873,539 | 28.13 | 6.5 |
| 9 | `address_rare_word` | S2 | 3,939,786 | 715,980 | 21.53% | 57,500 | 1.73% | 2,861,355 | 1.44 | 20.1 |
| 9 | `address_rare_word` | S3 | 3,940,837 | 706,846 | 19.90% | 64,555 | 1.82% | 2,898,159 | 1.46 | 22.3 |

**Reading the table.**
- **Cheap, high-precision channels.** `address_exact` reaches 12.48% of S2 links
  for 0.26 candidates per query, at over 80% purity. The exact-name channels
  together cost 3.0 (S2) / 3.6 (S3) pairs per query and add 22.1% / 26.1% of links.
- **Recall carriers.** The combination channels provide most of the remaining
  recall (27.8–28.8 points from name pairs, 9.8–13.2 from name × number, 10.3–10.8
  from number × word), but they also provide about 90% of all candidates.
  `address_number_x_word` is the least efficient, at about 6 genuine links per
  1,000 added pairs.
- **Every channel adds recall.** Even the last and cheapest, `address_rare_word`,
  adds 1.7–1.8% of links.

## Bucket caps and frequency controls

"Queries with ≥1 key above cap" counts queries that had at least one key skipped.
For single-token channels this is nearly every query, because each query contains
some common token such as a city name; that is the intended behavior. The last two
columns count **missed** links (not recovered by any channel) whose query shares a
key with the target in that channel, but only a key that was over the cap or that
fell outside the key limit.

| Channel | Cap / key limit | Source | Queries with ≥1 key above cap | Queries hitting key limit | Largest expanded bucket | Missed links sharing a key only above cap | Missed links lost to key limit |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `address_exact` | 50 / all | S2 | 0 | 0 | 19 | 0 | 0 |
| `address_exact` | 50 / all | S3 | 0 | 0 | 6 | 0 | 0 |
| `name_exact` | 50 / all | S2 | 45,425 | 0 | 50 | 4,896 | 0 |
| `name_exact` | 50 / all | S3 | 46,440 | 0 | 50 | 4,682 | 0 |
| `name_folded` | 50 / all | S2 | 55,497 | 0 | 50 | 5,814 | 0 |
| `name_folded` | 50 / all | S3 | 57,866 | 0 | 50 | 5,526 | 0 |
| `name_folded_compact` | 50 / all | S2 | 61,880 | 0 | 50 | 6,127 | 0 |
| `name_folded_compact` | 50 / all | S3 | 64,596 | 0 | 50 | 5,825 | 0 |
| `name_rare_token` | 30 / 1 | S2 | 1,983,940 | 15,148 | 30 | 101,532 | 20 |
| `name_rare_token` | 30 / 1 | S3 | 1,983,967 | 14,157 | 30 | 133,784 | 21 |
| `name_token_pair` | 50 / 8 | S2 | 1,656,966 | 37,447 | 50 | 79,985 | 27 |
| `name_token_pair` | 50 / 8 | S3 | 1,670,615 | 36,858 | 50 | 110,238 | 15 |
| `name_token_x_address_number` | 50 / 8 | S2 | 1,100,603 | 18,385 | 50 | 4,308 | 5 |
| `name_token_x_address_number` | 50 / 8 | S3 | 1,132,414 | 19,673 | 50 | 17,844 | 12 |
| `address_number_x_word` | 50 / 8 | S2 | 1,294,864 | 172,201 | 50 | 45,158 | 3,645 |
| `address_number_x_word` | 50 / 8 | S3 | 1,249,474 | 209,746 | 50 | 49,600 | 3,968 |
| `address_rare_word` | 20 / 2 | S2 | 1,986,411 | 24,106 | 20 | 178,380 | 452 |
| `address_rare_word` | 20 / 2 | S3 | 1,986,411 | 25,297 | 20 | 183,346 | 766 |

**Where the limits bite.**
- **Frequency caps, not key limits.** A missed link can be counted under several
  channels. Key limits cost at most 3,645 (S2) / 3,968 (S3) missed links in any
  one channel (`address_number_x_word`); `address_rare_word` loses 452 / 766, and
  every other channel 27 or fewer.
- **Exact names.** The exact-name caps of 50 skip 45–65K queries per source.
  Between 4,682 and 6,127 missed links per channel and source shared such a capped
  exact name.
- **No key shared at all.** Of the 426,573 missed links, only **1,674** share no
  key with their target in any channel: 115 in S2 and 1,559 in S3. Every other miss
  shares at least one key and is lost to limits.

**Index sizes** (after the keep-200 filter):

| Channel | S2 postings | S2 MiB | S3 postings | S3 MiB |
| --- | ---: | ---: | ---: | ---: |
| `address_exact` | 4,865,649 | 56 | 5,109,687 | 58 |
| `name_exact` | 5,026,429 | 58 | 5,276,999 | 60 |
| `name_folded` | 5,025,725 | 58 | 5,275,680 | 60 |
| `name_folded_compact` | 5,023,825 | 57 | 5,274,481 | 60 |
| `name_rare_token` | 3,312,420 | 38 | 3,620,265 | 42 |
| `name_token_pair` | 16,471,462 | 189 | 18,642,505 | 214 |
| `name_token_x_address_number` | 17,857,780 | 204 | 18,988,926 | 217 |
| `address_number_x_word` | 29,736,981 | 341 | 30,268,785 | 347 |
| `address_rare_word` | 4,086,833 | 47 | 3,904,583 | 45 |

## Failure modes of missed links (full development)

| Diagnostic | S2 | S3 |
| --- | ---: | ---: |
| Missed links | 205,937 | 220,636 |
| Share some key, but only above caps or limits | 205,822 | 219,077 |
| Share no key in any channel | 115 | 1,559 |
| Share any address word (often a city or state) | 178,832 | 184,112 |
| Share any name token | 101,552 | 133,805 |
| Share any address number | 48,917 | 55,227 |
| Target address empty | 26,986 | 34,605 |

The dominant failure mode is **generic evidence**: the only shared keys are
common name tokens or locality words, which are too frequent to expand under the
caps. Other failures:
- **Empty target addresses** (13–16% of misses) leave only the name.
- **Unshared names.** Indic-script target names can never share a token with
  all-ASCII S1 names, as Phase 2 showed. Name matching for them depends on address
  keys, and name keys fail entirely where their addresses are also generic.
- **No shared key at all.** These 1,674 links need approximate string retrieval:
  typos in every token, or transliteration.

## Country isolation

Country is part of every key. The full run found **0 cross-country candidates**
and **0 development links whose S1 and target countries differ** in either source,
so the country restriction lost no development link. Tests cover an unseen label
(`Atlantis`) and `France`.

## Resources

| Item | Value |
| --- | --- |
| Key stores (S1 / S2 / S3) | 333 / 729 / 760 MiB (1.78 GiB total) |
| Indexes (S2 / S3) | 1,047 / 1,104 MiB (2.10 GiB total) |
| Steady-state derived artifacts | 3.9 GiB |
| Store build, one-time (6 workers) | 34 s / 85 s / 86 s |
| Index build, one-time (cached) | ≈64 s (S2), ≈102 s (S3) |
| Full development evaluation (indexes cached) | 325 s total (S2 pass 128 s, S3 pass 129 s) |
| Sample evaluation | 272.5 s, including the one-time index build (≈166 s) |
| Largest transient build files | ≈0.53 GiB of partition files while building `address_number_x_word` |

**Disk.** A first index version kept every posting, including buckets no channel
can ever expand. It reached 3.8 GiB of indexes and 5.6 GiB of blocking
artifacts in total, leaving 5.7 GB free, which was too close for this workstation.
Filtering at `index_keep = 200` cut the indexes to 2.1 GiB with **identical
metrics**; the sample report was compared field by field. Every store and index
build checks free space against an estimate and a 3 GiB reserve, and refuses to
start otherwise. Builds write to a staging directory that is removed on failure.
Disk free was 7.35 GiB before and after the full run.

**Memory.** A first design built all channel indexes in memory and was killed by
the kernel's out-of-memory killer at about 3.9 GB anonymous RSS. The external,
memory-mapped design keeps anonymous memory bounded. Runs are executed in a
2.8 GB `systemd-run` memory scope. The reported `ru_maxrss` of 3.2 GiB includes
reclaimable memory-mapped index and store pages.

## Official output compatibility

Nothing has been written for test yet.
- **Scale.** At the development mean of 141 candidates per entity, a test
  `candidate_pairs.tsv` would list about 244M IDs, roughly 3 GB. The candidate file
  must be the exact set fed to the matcher.
- **Needed first.** A test writer therefore needs per-source spill files merged
  per batch (the current in-memory writer is for the 2,000-row debug sample only).
  More importantly, a pruning stage (see below) should come first.

## Weaknesses

- **Volume, not recall, is the main weakness.** About 90% of candidates come from
  the three combination channels, whose added pairs are only 0.6–3.3% genuine.
  Downstream scoring of 280M development pairs, or 244M test pairs, is expensive.
- **India is weaker on both counts:** 92.1% recall at 153 candidates, against the
  US at 94.9% and 133.
- **Cap-limited misses dominate** (99.6% of misses). Generic names combined with
  common locality words are the hardest case.
- **No-shared-key misses are few** (1,674), concentrated in S3, and need
  approximate retrieval or transliteration.
- **Channel order is not a ranking.** Incremental accounting depends on channel
  order; standalone figures are order-independent.

## Recommendation for Phase 3B

Approximate retrieval is **not** the main need. With deterministic keys alone,
99.6% of misses already share a key with their target. The priorities, measured on
development only as before, are:

1. **Candidate pruning / ranking.** Add a cheap per-source score over the existing
   union, for example the number and rarity of shared keys and the channel
   bitmask. Keep the top-K per source, and measure recall against K. This is the
   largest lever on volume, and it determines what the official candidate file can
   contain.
2. **Recover cap-limited links with more specific conjunctions** rather than larger
   caps. Examples:
   - a generic name token × a locality word, built with the keep filter;
   - three-token name keys;
   - name pair × number.

   Report each channel's incremental recall per added pair, as here.
3. **Only then, targeted approximate retrieval.** Character n-gram / TF-IDF
   retrieval would cover typo-in-every-token names, the 1,674 no-shared-key links,
   and parts of the Indic-script names. It should run **only for queries whose
   deterministic candidate set is small or has no high-precision hit**, not for
   every query.
4. **Measure a transliteration view separately.** Indic-script target names
   (427K in S2 and 251K in S3, from Phase 2) could form an auxiliary
   transliterated view, measured as its own channel.
