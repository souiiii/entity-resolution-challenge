# Phase 8 — Candidate Recall Recovery

Phase 8 changes **candidate generation only**. The Phase 6 LightGBM model, its threshold (0.660),
the frozen split, normalization, `blocking_v2.json`, `matcher_v1.json`, `supervised_v1.json` and
`decision_v1.json` are unchanged. There was no retraining, threshold work, test inference or
submission. All design work used **development labels only**, on the existing 40,170-query hash
sample. The selected design was then run once on full development and once on the frozen
validation split.

**Starting point.** Branch `phase8-candidate-recall` at
`dbdf4401fe3a0a6a828d73def2131172785b8fb6`, the same commit as `main`.

## Reproduction

```bash
python scripts/analyze_candidate_recall.py                         # loss accounting (sample)
python scripts/run_candidate_recovery.py --collect-sample          # temporary deep pool (~1.1 GB)
python scripts/run_candidate_recovery.py --evaluate-pool           # designs 1 and 2 (sample)
python scripts/run_candidate_recovery.py --evaluate --partition sample
python scripts/analyze_candidate_recall.py --residual              # remaining losses (sample)
python scripts/run_candidate_recovery.py --evaluate --partition development
python scripts/run_candidate_recovery.py --evaluate --partition validation   # refuses a second run
python -m pytest -q
```

Reports are written to ignored `data/interim/candidates/`. The code lives in
`src/blocking/recovery.py`, `src/matching/final_candidates.py` and the two scripts above. The frozen
policy is `experiments/configs/blocking_v3.json`, sha256
`42ab3b48abfacb758d77ed9172d68791832a4cb9ed649da76fe00250c8c9ca65`.

## Starting baseline

| Candidate set | Split | True-link recall | Oracle Macro F0.5 ceiling | Mean candidates / query |
| --- | --- | ---: | ---: | ---: |
| Phase 3B (aggregate rank, K = 50 per source) | development sample | 94.855% | 0.98057 | 86.00 |
| Phase 5 final screen | development sample | 94.024% | 0.97737 | 24.73 |
| Phase 5 final screen | full development | 93.988% | 0.97749 | 24.79 |
| Phase 5 final screen | frozen validation | 94.014% | 0.97747 | 24.78 |

Every Phase 8 run re-measures Phase 5 inside the same pass (provenance bit `phase5`). The
full-development and validation Phase 5 rows reproduce earlier reports exactly: 5,462,251
validation pairs and 94.014% recall.

## Loss accounting — where the missing ~6% go (development sample)

The accounting replays each stage exactly:
- the Phase 3B channel union;
- index keep limits;
- per-channel frequency caps and key limits;
- the aggregate rank with K = 50;
- the Phase 5 prefilter, using each link's real Phase 4 evidence.

It reproduces the known sample recalls exactly: union 95.5917%, top-50 94.8550%, Phase 5
94.0239%, which is 130,555 of 138,853 links.

Each missed link gets its **least restrictive** loss point:
- if it reached the union, the rank cut or prefilter;
- otherwise, the best shared key across all channels:
  - within the cap but not among the query's rarest keys → **key limit**;
  - above the cap but in a stored bucket → **frequency cap**;
  - only in buckets above the index keep limit → **truncated**;
  - none → **no shared key**.

| Category | Missed links | Share of missed | Share of all links | S2 / S3 | India / US |
| --- | ---: | ---: | ---: | --- | --- |
| A — no shared deterministic key | 32 | 0.39% | 0.02% | 0 / 32 | 9 / 23 |
| B — all shared buckets truncated (> index keep) | 2,736 | 32.97% | 1.97% | 1,412 / 1,324 | 1,615 / 1,121 |
| C — retrievable bucket above channel frequency cap | 2,936 | 35.38% | 2.11% | 1,385 / 1,551 | 1,882 / 1,054 |
| C — eligible key dropped by channel key limit | 417 | 5.03% | 0.30% | 163 / 254 | 314 / 103 |
| D — in union, aggregate rank > 50 | 1,023 | 12.33% | 0.74% | 477 / 546 | 461 / 562 |
| E — in top 50, removed by Phase 5 prefilter | 1,154 | 13.91% | 0.83% | 482 / 672 | 299 / 855 |
| F — other | 0 | 0% | 0% | — | — |
| **Total** | **8,298** | 100% | 5.98% | | |

**The historical hypothesis holds.** 99.6% of missed links share at least one deterministic key;
only 32 do not.

**What each category looks like.** Similarities are from Phase 4 text; "shared name tokens" are
hashed tokens.

| Category | Script mismatch | Number state | Name similarity (median) | Address similarity (median) | Recovery channels and buckets |
| --- | ---: | --- | ---: | ---: | --- |
| B | 1,105 (40%) | unavailable 51%, agree 28%, conflict 18% | 0.667 | 0.881 | Best shared bucket 51–200: 65 links; > 200: 2,671. `address_rare_word` p50 5,040, `name_token_pair` p50 1,484, `address_number_x_word` p50 740 |
| C frequency cap | 999 (34%) | unavailable 55%, agree 30% | 0.896 | 0.911 | Buckets 11–50: 763; 51–200: 2,173. Lifting a cap to the keep limit reaches: `address_rare_word` 1,017, `name_token_pair` 1,013, `address_number_x_word` 978, `name_x_address_word` 649, others ≤ 191 |
| C key limit | 150 (36%) | unavailable 45%, agree 37% | 0.867 | 0.891 | Mostly `address_number_x_word` and `name_x_address_word` |
| D | 187 (18%) | agree 58% | 0.941 | 0.889 | Rank p50 62, p90 114. 81% have single-family support |
| E | 78 (7%) | **conflict 92%** | 0.965 | 0.870 | Rank p50 12. `name_x_address_word` alone: 474 |

**Strongest opportunities.**
- **E is a precise fix.** Reformatted or truncated house numbers produce conflicts on pairs whose
  names and addresses are otherwise near-identical.
- **D is cheap to recover from the existing union.**
- **C is the largest recoverable class.** It needs no new index, only deeper retrieval from buckets
  already stored.
- **B would need buckets that are not stored at all**, and has the weakest supporting evidence.

## Designs tested (development sample; three designs, no fourth)

The same explainable **cheap score** is used throughout. It is computed from hashed store tokens,
reads no text and builds no Phase 4 matrix:

```text
score = name-token containment + address-word containment
        + number term (+0.5 agree, -0.5 conflict) + 1[exact folded name or exact address]
        + min(blocking rarity, 1)
```

Missing containments count as 0.

The designs:
- **Design 1:** keep the Phase 3B union; the final set is the top M per query and source by cheap
  score.
- **Design 2:** deep retrieval, then the top M by cheap score. Deep retrieval means every channel
  on its existing cached index, with its frequency cap lifted to the index keep limit (200; 50 for
  `name_x_address_word`) and no key limit. It is a superset of the union, about 888 candidates per
  query, and contains 98.01% of links.
- **Design 3:** targeted additive rescue on top of the Phase 5 screen:
  - **conflict rescue:** a top-50 candidate with a number conflict, text name ≥ 0.8 and text
    address ≥ 0.6;
  - **rank rescue:** a union candidate ranked above 50 whose cheap score is ≥ 1.5, or the query's
    single best by score;
  - **deep rescue:** the query's `m` best deep-only candidates by cheap score, for each source.

### Pareto table (development sample)

Phase 5 is the reference for the gain columns.

| Design / point | Recall | Gain | Oracle | Pairs | Mean | p99 | +true / 1,000 added pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 5 (reference) | 94.024% | — | 0.97737 | 993,372 | 24.73 | 68 | — |
| Phase 3B union (upper reference) | 95.59% | +1.57 | 0.98346 | 5,962,708 | 148.44 | 401 | 0.44 |
| Deep pool (upper reference) | 98.01% | +3.98 | 0.99321 | 35,667,649 | 887.92 | 2,910 | 0.16 |
| D1 top 12 | 93.81% | −0.21 | 0.97745 | 950,004 | 23.65 | 24 | — |
| D1 top 20 | 94.64% | +0.62 | 0.98024 | 1,551,510 | 38.62 | 40 | 1.54 |
| D1 top 25 | 94.87% | +0.84 | 0.98099 | 1,909,401 | 47.53 | 50 | 1.28 |
| D2 top 12 | 93.90% | −0.12 | 0.97824 | 963,880 | 24.00 | 24 | — |
| D2 top 20 | 94.97% | +0.94 | 0.98206 | 1,605,982 | 39.98 | 40 | 2.14 |
| D2 top 25 | 95.29% | +1.26 | 0.98311 | 2,006,982 | 49.96 | 50 | 1.73 |
| D3 Phase 5 + conflict rescue | 94.626% | +0.60 | 0.97978 | 1,006,610 | 25.06 | 69 | **63.15** |
| D3 + rank rescue | 95.053% | +1.03 | 0.98158 | 1,065,640 | 26.53 | 71 | 19.77 |
| D3 + deep 1 | 95.569% | +1.55 | 0.98386 | 1,145,623 | 28.52 | 73 | 14.09 |
| D3 + deep 3 | 95.822% | +1.80 | 0.98489 | 1,305,531 | 32.50 | 77 | 8.00 |
| D3 + deep 5 | 95.971% | +1.95 | 0.98546 | 1,465,320 | 36.48 | 81 | 5.73 |
| **D3 + deep 8 (selected)** | **96.108%** | **+2.08** | **0.98598** | 1,704,717 | **42.44** | 87 | 4.07 |
| D3 + deep 10 | 96.187% | +2.16 | 0.98636 | 1,864,132 | 46.41 | 91 | 3.45 |
| D3 + deep 15 | 96.325% | +2.30 | 0.98694 | 2,261,896 | 56.31 | 101 | 2.52 |

Designs 1 and 2 are reproducible from the pool (`--evaluate-pool`); Design 3 rows come from the
production generator. Every Design 3 point keeps zero zero-candidate queries.

**What the designs showed.**
- **Designs 1 and 2 fail.** At equal size, cheap-score ranking loses to the text-based Phase 5
  screen. Hashed tokens cannot see typos, scripts or partial matches, and a fixed top M wastes
  budget on queries with obvious answers.
- **Design 3 dominates** at every size.
- **Two refinements inside Design 3 were rejected.** IDF-weighted overlap, which weights shared
  tokens by log(N / (1 + f)) from the existing single-token indexes, was slightly **worse** on a
  10,000-query subset: top-1 deep rescue 183 against 188 true links. Restricting deep retrieval to
  fewer channels saved at most about 14% of the pool while losing about 4% of rescues, because
  every channel contributes.

**No fourth design.** The only unaddressed class is B. A fourth design would require stored
postings for buckets above 200. Of the B links:
- 875 have a best shared bucket ≤ 500, and 1,430 have one ≤ 1,000;
- they have the weakest evidence: median name similarity 0.667, 40% script mismatch, 51% with no
  number evidence.

C links are easier than that (buckets ≤ 200, stronger evidence), yet their efficient rescue
already falls to about 4 true links per 1,000 added pairs at m = 8. A B design would be larger and
less efficient, and the diagnostics expose no genuinely separate high-value mechanism. A (32
links) is negligible, so no approximate string retrieval was built.

## Selected design — `blocking-v3`

Design 3 at `deep_per_query_source = 8`:

| Stage | Rule | Evidence used |
| --- | --- | --- |
| Base | Phase 3B union and aggregate rank (unchanged) | blocking |
| Phase 5 screen | unchanged `matcher_v1` prefilter on the top 50 | Phase 4 features (top 50 only, as before) |
| Conflict rescue | top 50, number conflict and name ≥ 0.8 and address ≥ 0.6 | same Phase 4 evidence |
| Rank rescue | rank > 50 and (cheap score ≥ 1.5, or best cheap score of the query and source) | hashed tokens and blocking |
| Deep rescue | best 8 deep-only candidates per query and source by cheap score; ties by smallest shared-key frequency, then target ID | hashed tokens and deep retrieval |

**Final candidate semantics.** `iter_final_candidates` yields **exactly** the union of the five
stages above, per source, in query order and then binary target-ID order. Each candidate carries
provenance flags. This set is what the next matcher must score, and what `candidate_pairs.tsv`
must contain. Nothing is scored by a matching model before selection.

Phase 4 features are still computed only for the top 50, which is today's cost. Phase 9 will
compute its features for the final set (about 42 per query). A regression test shows the `phase5`
provenance equals the existing `collect_candidates` + `screen` output exactly, and the final set
always contains the Phase 5 set: in the sample, no Phase 5 link was lost.

**Freeze.** The policy was frozen from the sample before any full-development or validation run.
- `blocking_v3.json` sha256 `42ab3b48…` is unchanged throughout.
- `final_candidates.py` sha256 `4c36e65d…` is unchanged throughout.
- `recovery.py` changed once after freezing, from `83e9ca94…` to `709d34c4…`. The change is a
  clear `ValueError` in `deepen` for channels without an index keep limit; every `blocking_v3`
  channel has one, so behavior is identical.
- Evaluation-only code later gained a per-link export for the sample; the rerun reproduced every
  stage metric exactly.

## Full-development confirmation (1,986,411 queries, one run)

| Metric | Phase 5 | Phase 8 (`deep8`) | Sample `deep8` |
| --- | ---: | ---: | ---: |
| Recall | 93.988% | **96.054%** (+2.07 pp) | 96.108% (+2.08 pp) |
| Oracle Macro F0.5 ceiling | 0.97749 | **0.98592** | 0.98598 |
| Pairs | 49,237,405 | 84,412,160 | 1,704,717 |
| Mean / median / p90 / p95 / p99 / max | 24.79 / 21 / 43 / 51 / 68 / 100 | 42.49 / 39 / 62 / 70 / 87 / 185 | 42.44 / 39 / 62 / 70 / 87 / 123 |
| Zero-candidate queries | 23 | 1 | 0 |
| Entities all / some / no truth recovered (of 1,875,524) | 1,570,932 / 1,861,214 / 14,310 | 1,657,268 / 1,867,473 / 8,051 | — |
| S2 / S3 recall | 94.05% / 93.93% | 96.03% / 96.08% | 96.15% / 96.07% |
| India / US recall | 91.68% / 95.53% | 93.84% / 97.53% | 93.98% / 97.53% |

142,097 more true links for 35,174,755 more pairs (4.04 per 1,000). The recall gain, candidate
counts, S2/S3 balance and India/US gains (+2.16 and +2.00 pp) all match the sample.

**Maximum candidate count.** The rank rescue's score ≥ 1.5 path has no per-query cap, so the
maximum reached 185 for one query (p99 87). This was frozen behavior and is reported as observed.

## Frozen validation (220,410 queries, one run, candidate-only)

| Metric | Old Phase 5 | Phase 8 `blocking-v3` |
| --- | ---: | ---: |
| Recall | 94.014% | **96.101%** (+2.09 pp) |
| Oracle Macro F0.5 ceiling | 0.97747 | **0.98606** |
| Pairs | 5,462,251 | 9,363,000 |
| Mean / median / p90 / p95 / p99 / max | 24.78 / 21 / 43 / 51 / 67 / 100 | **42.48** / 39 / 62 / 70 / 86 / 178 |
| Zero-candidate queries | 4 | 1 |
| Entities all / some / no truth recovered (of 208,050) | 174,467 / 206,437 / 1,613 | 184,157 / 207,154 / 896 |
| S2 / S3 recall | 94.08% / 93.95% | 96.07% / 96.13% |
| India / US recall | 91.77% / 95.51% | 93.94% / 97.54% |
| Added true links / added pairs | — | 15,894 / 3,900,749 (4.07 per 1,000) |

Validation matches development within 0.05 points. The candidate design was not altered after
this run. The report and its start marker prevent a second attempt.

## Residual missed links (development sample, after Phase 8)

| Category | Phase 5 missed | Phase 8 missed | Recovered | Share of remaining |
| --- | ---: | ---: | ---: | ---: |
| A — no shared key | 32 | 32 | 0 | 0.6% |
| B — truncated buckets | 2,736 | 2,736 | 0 | **50.6%** |
| C — frequency cap | 2,936 | 1,792 | 1,144 | 33.2% |
| C — key limit | 417 | 96 | 321 | 1.8% |
| D — rank cut | 1,023 | 430 | 593 | 8.0% |
| E — prefilter | 1,154 | 318 | 836 | 5.9% |
| **Total** | 8,298 | **5,404** | 2,894 | 100% |

## Success criteria

| Criterion | Target (minimum / stretch) | Result |
| --- | --- | --- |
| Recall gain | ≥ +2.0 pp / reach ≥ 98.5% | **Met minimum** (+2.09 pp validation; 96.10%). Stretch not met |
| Oracle Macro F0.5 | ≥ 0.990 / ≥ 0.995 | **Not met** (0.98606) |
| Mean final candidates / query | ≤ 50 / ≤ 40 | **Met minimum** (42.48). Stretch not met |

**Why 0.990 is out of reach with existing deterministic evidence.**
- **The oracle curve is flat against candidate count.** The whole 888-per-query deep pool (98.0%
  recall) reaches only 0.99321, and the 148-per-query Phase 3B union 0.98346.
- **Every remaining recoverable link is hard to rank cheaply.** The median cheap-score rank of
  deep-only true links is 13 and the 90th percentile 154. Half share no hashed name token, and 53%
  have no number evidence on one side. Reaching 0.990 needs roughly 97% recall; covering the
  remaining C links within 50 candidates would need discrimination hashed tokens do not provide.
- **Half of the remaining loss (B) sits in buckets that are not stored.**

## Resources

| Run | Runtime | Peak RSS (includes mapped files) | Minimum free disk | Minimum available memory |
| --- | ---: | ---: | ---: | ---: |
| Loss accounting (sample) | 207 s | 4,983 MiB | 16.87 GiB | — |
| Deep pool collection (sample, temporary) | 461 s | 5,334 MiB | 16.79 GiB | 4,803 MiB |
| Design 3 production run (sample) | 317 s | 5,736 MiB | 15.52 GiB | 5,520 MiB |
| Full development | 8,168 s (2.3 h) | 6,480 MiB | **15.21 GiB** | 3,980 MiB |
| Validation | 905 s | 5,890 MiB | 15.21 GiB | 5,975 MiB |

- **No new index was built.** Deep retrieval reuses the existing Phase 3B indexes, and no raw data
  was duplicated.
- **The only large temporary artifact** was the 1.1 GB sample pool, deleted after the design
  study. The retained reports and link tables total 15 MB.
- **Guards.** Heavy runners disable core dumps and refuse to continue below 10.5 GiB free disk or
  1,024 MiB available memory.

## Implications for Phase 9

1. **Retrain on the new distribution.** Phase 9 must retrain LightGBM on `blocking-v3` final
   candidates. The existing model saw only Phase 5 screen candidates, so its validation score is
   not evidence for the new set.
2. **Reuse the rescue signals.** The rescued candidates come with provenance flags, cheap-score
   components, deep key-hit counts and blocking rarity. These are natural matcher features; the
   new rescues carry different, and often weaker, evidence than Phase 5 candidates.
3. **Treat the ceiling as the target.** Oracle rises from 0.97747 to 0.98606, about +0.0086 of
   headroom. The matcher reached 0.925 of 0.977 on the old set; Phase 9 must preserve precision
   while using the extra recall, especially for India (93.9% candidate recall).
4. **Plan for the cost.** Phase 9 will score about 42 candidates per query instead of about 25,
   roughly 1.7× the pairs. Full-development candidate generation takes about 2.3 h here; features
   must stream.
5. **Next candidate lever.** B links, with only very common shared keys, are now half of the
   remaining loss. Reaching them needs stored postings above bucket size 200 **plus** a
   discriminative ranker, most likely a trained cheap ranker or the Phase 9 model applied as a
   *declared* second-stage filter whose input set becomes the official candidate set. That is a
   decision for after Phase 9 measures how much of the new headroom it converts.
