# Phase 4 — Pairwise Feature Engineering

Phase 4 builds a deterministic, batch-wise pairwise feature extractor on top of the
frozen Phase 3B candidates, and validates the features on the development sample.
It has no matching decisions, thresholds, classifiers, approximate retrieval, or
validation/test predictions. Blocking (`blocking_v2.json`: aggregate ranking, K=50
per source), normalization (`canonical-v1`), country logic and the frozen split
(`1dd67d77fee74aa0b64477e10f09377ac22854f5eb13780f3aacc4baf94e3645`) are unchanged.

**Validation labels were not used.** Positive and negative pairs come only from
development Source 1 labels, through the existing development-only truth loader.
Every measurement below is from the existing label-independent 2% development hash
sample.

## Reproduction

```bash
python scripts/analyze_features.py --limit 2000   # smoke run
python scripts/analyze_features.py                # the one development-sample analysis
python -m pytest -q
```

The report is written to ignored `data/interim/features/features-v1_sample_report.json`.
The configuration is in `experiments/configs/features_v1.json`. RapidFuzz (3.14.6) was
already declared in `requirements.txt` but missing from the local environment, so it
was installed; no new dependency was added.

## Architecture

- **Text access** (`src/features/text.py`). The blocking stores hold only hashes, so
  text is read from the **unchanged raw TSVs** through one int64 byte offset per
  record, memory-mapped.
  - The offsets are built once. Every one of the 12.5M IDs is checked against the
    blocking store, in a staging directory that is removed on failure, and the
    build is refused if it would breach the 10 GiB free-space reserve.
  - Every read re-verifies the entity ID.
  - The only new persistent artifact is 100,216,344 bytes (95.6 MiB) of offsets.
    Free space stayed at 23.69 GiB. No normalized text corpus and no pair-level
    feature matrix were written.
- **Candidates** (`src/features/extract.py`). `select_candidates` runs the unchanged
  Phase 3B union, `candidate_ranks(..., 'aggregate')` and the K=50 cut, and keeps
  each candidate's rank. A test checks that it selects exactly what
  `prune_candidates` selects.
  - The sample reproduced the Phase 3B sample exactly: 3,454,436 pairs and 131,709
    recovered links (94.8550% of 138,853).
- **Features** (`src/features/pairwise.py`). Each record's Phase 2 canonical views
  are computed once per batch: canonical name, folded name, folded-compact name,
  token sets, script flag, canonical address, and integer-equivalent address
  numbers. `extract_pair_features` then returns a `float32` matrix whose columns
  follow `feature_names(channels)`.
- **Streaming.** `iter_features` streams one target source and one 5,000-query
  batch at a time. Blocking and text reads run in the main process; views and
  features run in six ordered worker processes. The output is deterministic and
  independent of target file order and of the number of workers (both tested).
  Future matching calls this same path.

`NaN` means *undefined*, never zero: a text similarity when either side is empty,
number Jaccard when neither side has a number, or a containment when the relevant
side has no number. Country only enters through the blocking keys; no feature
depends on a specific country label. Tests cover `France` and an unseen label.

## Feature schema (42 columns, fixed order)

| Group | Features | Rationale |
| --- | --- | --- |
| Source | `source_is_s3` | S2 and S3 have different noise; later stages may need to condition on source |
| Blocking evidence | `blocking_rank`, `blocking_support_channels`, `blocking_support_families`, `blocking_exact_name`, `blocking_min_key_frequency`, `blocking_aggregate_rarity` | Phase 3B evidence exactly as used for ranking. Rank is an ordinal position, not a probability |
| Channel indicators | `channel_<name>` for all 10 `blocking_v2` channels, in configuration order | Which specific keys matched; `channel_address_exact` is the exact-address support |
| Name | canonical / folded / folded-compact exact equality; `name_char_similarity` (RapidFuzz normalized Indel ratio on folded names); `name_token_set_similarity` (RapidFuzz token-set ratio); folded-token intersection, Jaccard and containment (overlap coefficient); `name_length_ratio` (shorter / longer); `name_script_mismatch` | Covers exact agreement, character noise, reordering and suffix changes, token overlap, and unmatched scripts |
| Address | canonical exact equality; char and token-set similarity; token intersection, Jaccard and containment; length ratio | The same families on canonical addresses |
| Address numbers | query and target number counts; shared count; number Jaccard; all query numbers in target; all target numbers in query; set equality | Integer-equivalent (`015115` ≡ `15115`) house, plot and floor numbers |
| Missingness | `target_address_empty` | Empty target addresses make address evidence impossible |

**Removed after the diagnostic run** (redundant or constant; nothing was tuned):
- `blocking_exact_address` duplicated `channel_address_exact` exactly.
- `query_name_empty`, `target_name_empty` and `query_address_empty` were 0 for
  every pair. Phase 1 showed that names are never empty in any file and Source 1
  addresses never empty in train or test.
- `address_both_present` was then exactly `1 − target_address_empty`.

Removing these five columns did not change the values of any remaining column.

## Sample pair counts

| Group | Pairs | Positive | Negative | Positive rate |
| --- | ---: | ---: | ---: | ---: |
| All | 3,454,436 | 131,709 | 3,322,727 | 3.81% |
| S2 | 1,741,207 | 63,825 | 1,677,382 | 3.67% |
| S3 | 1,713,229 | 67,884 | 1,645,345 | 3.96% |
| India | 1,372,422 | 51,251 | 1,321,171 | 3.73% |
| US | 2,082,014 | 80,458 | 2,001,556 | 3.86% |

There are 40,170 queries, 67,262 S2 links and 71,591 S3 links.

## Diagnostics (all sample pairs)

AUC is the univariate Mann–Whitney probability that a positive pair scores above a
negative one, computed over defined values only; coverage is reported separately.
Values below 0.5 mean that lower is better, as for rank and frequency. The AUC is a
diagnostic of separation **among blocked candidates**. It is not a model score,
and correlated features are not additive.

| Feature | Coverage + | Coverage − | Mean + | Mean − | p50 + | p50 − | Nonzero + | Nonzero − | AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `source_is_s3` | 1.000 | 1.000 | 0.515 | 0.495 | 1.000 | 0.000 | 0.515 | 0.495 | 0.510 |
| `blocking_rank` | 1.000 | 1.000 | 4.026 | 24.548 | 2.000 | 24.000 | 1.000 | 1.000 | 0.062 |
| `blocking_support_channels` | 1.000 | 1.000 | 4.082 | 1.179 | 4.000 | 1.000 | 1.000 | 1.000 | 0.915 |
| `blocking_support_families` | 1.000 | 1.000 | 3.605 | 1.067 | 4.000 | 1.000 | 1.000 | 1.000 | 0.927 |
| `blocking_exact_name` | 1.000 | 1.000 | 0.272 | 0.069 | 0.000 | 0.000 | 0.272 | 0.069 | 0.602 |
| `blocking_min_key_frequency` | 1.000 | 1.000 | 2.383 | 16.710 | 2.000 | 14.000 | 1.000 | 1.000 | 0.052 |
| `blocking_aggregate_rarity` | 1.000 | 1.000 | 3.753 | 0.172 | 3.047 | 0.083 | 1.000 | 1.000 | 0.975 |
| `channel_address_exact` | 1.000 | 1.000 | 0.087 | 0.001 | 0.000 | 0.000 | 0.087 | 0.001 | 0.543 |
| `channel_name_exact` | 1.000 | 1.000 | 0.221 | 0.059 | 0.000 | 0.000 | 0.221 | 0.059 | 0.581 |
| `channel_name_folded` | 1.000 | 1.000 | 0.259 | 0.061 | 0.000 | 0.000 | 0.259 | 0.061 | 0.599 |
| `channel_name_folded_compact` | 1.000 | 1.000 | 0.270 | 0.060 | 0.000 | 0.000 | 0.270 | 0.060 | 0.605 |
| `channel_name_rare_token` | 1.000 | 1.000 | 0.158 | 0.045 | 0.000 | 0.000 | 0.158 | 0.045 | 0.557 |
| `channel_name_token_pair` | 1.000 | 1.000 | 0.650 | 0.232 | 1.000 | 0.000 | 0.650 | 0.232 | 0.709 |
| `channel_name_token_x_address_number` | 1.000 | 1.000 | 0.676 | 0.219 | 1.000 | 0.000 | 0.676 | 0.219 | 0.728 |
| `channel_address_number_x_word` | 1.000 | 1.000 | 0.780 | 0.364 | 1.000 | 0.000 | 0.780 | 0.364 | 0.708 |
| `channel_address_rare_word` | 1.000 | 1.000 | 0.219 | 0.036 | 0.000 | 0.000 | 0.219 | 0.036 | 0.591 |
| `channel_name_x_address_word` | 1.000 | 1.000 | 0.763 | 0.101 | 1.000 | 0.000 | 0.763 | 0.101 | 0.831 |
| `name_canonical_equal` | 1.000 | 1.000 | 0.229 | 0.059 | 0.000 | 0.000 | 0.229 | 0.059 | 0.585 |
| `name_folded_equal` | 1.000 | 1.000 | 0.270 | 0.067 | 0.000 | 0.000 | 0.270 | 0.067 | 0.601 |
| `name_folded_compact_equal` | 1.000 | 1.000 | 0.282 | 0.069 | 0.000 | 0.000 | 0.282 | 0.069 | 0.607 |
| `name_char_similarity` | 1.000 | 1.000 | 0.814 | 0.504 | 0.885 | 0.471 | 1.000 | 1.000 | 0.836 |
| `name_token_set_similarity` | 1.000 | 1.000 | 0.887 | 0.562 | 1.000 | 0.540 | 1.000 | 1.000 | 0.840 |
| `name_token_intersection` | 1.000 | 1.000 | 2.660 | 1.235 | 3.000 | 1.000 | 0.882 | 0.677 | 0.780 |
| `name_token_jaccard` | 1.000 | 1.000 | 0.664 | 0.244 | 0.667 | 0.167 | 0.882 | 0.677 | 0.822 |
| `name_token_containment` | 1.000 | 1.000 | 0.799 | 0.370 | 1.000 | 0.333 | 0.882 | 0.677 | 0.820 |
| `name_length_ratio` | 1.000 | 1.000 | 0.869 | 0.736 | 0.933 | 0.763 | 1.000 | 1.000 | 0.716 |
| `name_script_mismatch` | 1.000 | 1.000 | 0.057 | 0.032 | 0.000 | 0.000 | 0.057 | 0.032 | 0.512 |
| `address_canonical_equal` | 1.000 | 1.000 | 0.087 | 0.001 | 0.000 | 0.000 | 0.087 | 0.001 | 0.543 |
| `address_char_similarity` | 0.966 | 0.989 | 0.798 | 0.429 | 0.842 | 0.405 | 1.000 | 1.000 | 0.944 |
| `address_token_set_similarity` | 0.966 | 0.989 | 0.916 | 0.481 | 0.941 | 0.455 | 1.000 | 1.000 | 0.981 |
| `address_token_intersection` | 1.000 | 1.000 | 5.993 | 1.725 | 5.000 | 1.000 | 0.965 | 0.748 | 0.896 |
| `address_token_jaccard` | 0.966 | 0.989 | 0.642 | 0.128 | 0.667 | 0.100 | 1.000 | 0.756 | 0.972 |
| `address_token_containment` | 0.966 | 0.989 | 0.805 | 0.234 | 0.833 | 0.200 | 1.000 | 0.756 | 0.973 |
| `address_length_ratio` | 0.966 | 0.989 | 0.862 | 0.782 | 0.898 | 0.809 | 1.000 | 1.000 | 0.673 |
| `query_number_count` | 1.000 | 1.000 | 1.330 | 1.398 | 1.000 | 1.000 | 0.946 | 0.968 | 0.477 |
| `target_number_count` | 1.000 | 1.000 | 1.266 | 1.558 | 1.000 | 1.000 | 0.863 | 0.950 | 0.419 |
| `shared_number_count` | 1.000 | 1.000 | 1.088 | 0.592 | 1.000 | 1.000 | 0.808 | 0.571 | 0.668 |
| `number_jaccard` | 0.956 | 0.994 | 0.763 | 0.359 | 1.000 | 0.250 | 0.845 | 0.574 | 0.749 |
| `query_numbers_in_target` | 0.946 | 0.968 | 0.776 | 0.374 | 1.000 | 0.000 | 0.776 | 0.374 | 0.701 |
| `target_numbers_in_query` | 0.863 | 0.950 | 0.815 | 0.320 | 1.000 | 0.000 | 0.815 | 0.320 | 0.748 |
| `number_set_equal` | 0.956 | 0.994 | 0.681 | 0.224 | 1.000 | 0.000 | 0.681 | 0.224 | 0.729 |
| `target_address_empty` | 1.000 | 1.000 | 0.034 | 0.011 | 0.000 | 0.000 | 0.034 | 0.011 | 0.512 |

AUC by source and country:

| Feature | S2 | S3 | India | US | S2:India | S2:US | S3:India | S3:US |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `source_is_s3` | 0.500 | 0.500 | 0.510 | 0.510 | 0.500 | 0.500 | 0.500 | 0.500 |
| `blocking_rank` | 0.056 | 0.069 | 0.065 | 0.061 | 0.052 | 0.057 | 0.077 | 0.064 |
| `blocking_support_channels` | 0.917 | 0.914 | 0.890 | 0.932 | 0.892 | 0.933 | 0.888 | 0.930 |
| `blocking_support_families` | 0.928 | 0.927 | 0.900 | 0.944 | 0.899 | 0.946 | 0.901 | 0.943 |
| `blocking_exact_name` | 0.603 | 0.600 | 0.581 | 0.615 | 0.582 | 0.615 | 0.579 | 0.614 |
| `blocking_min_key_frequency` | 0.047 | 0.057 | 0.064 | 0.045 | 0.058 | 0.040 | 0.069 | 0.050 |
| `blocking_aggregate_rarity` | 0.978 | 0.973 | 0.962 | 0.983 | 0.966 | 0.985 | 0.958 | 0.982 |
| `channel_address_exact` | 0.565 | 0.522 | 0.540 | 0.545 | 0.559 | 0.569 | 0.522 | 0.523 |
| `channel_name_exact` | 0.582 | 0.580 | 0.567 | 0.590 | 0.569 | 0.590 | 0.565 | 0.589 |
| `channel_name_folded` | 0.599 | 0.599 | 0.579 | 0.611 | 0.580 | 0.611 | 0.578 | 0.612 |
| `channel_name_folded_compact` | 0.606 | 0.604 | 0.581 | 0.620 | 0.582 | 0.620 | 0.579 | 0.619 |
| `channel_name_rare_token` | 0.559 | 0.555 | 0.564 | 0.552 | 0.567 | 0.554 | 0.561 | 0.551 |
| `channel_name_token_pair` | 0.710 | 0.708 | 0.692 | 0.719 | 0.688 | 0.725 | 0.696 | 0.715 |
| `channel_name_token_x_address_number` | 0.720 | 0.736 | 0.679 | 0.759 | 0.670 | 0.752 | 0.688 | 0.766 |
| `channel_address_number_x_word` | 0.714 | 0.702 | 0.673 | 0.730 | 0.706 | 0.720 | 0.643 | 0.741 |
| `channel_address_rare_word` | 0.596 | 0.587 | 0.652 | 0.553 | 0.665 | 0.553 | 0.641 | 0.553 |
| `channel_name_x_address_word` | 0.840 | 0.822 | 0.783 | 0.862 | 0.795 | 0.870 | 0.772 | 0.854 |
| `name_canonical_equal` | 0.586 | 0.584 | 0.567 | 0.596 | 0.569 | 0.597 | 0.565 | 0.596 |
| `name_folded_equal` | 0.602 | 0.601 | 0.579 | 0.616 | 0.580 | 0.616 | 0.578 | 0.615 |
| `name_folded_compact_equal` | 0.608 | 0.605 | 0.581 | 0.622 | 0.582 | 0.623 | 0.579 | 0.622 |
| `name_char_similarity` | 0.833 | 0.839 | 0.764 | 0.882 | 0.747 | 0.887 | 0.780 | 0.876 |
| `name_token_set_similarity` | 0.831 | 0.849 | 0.772 | 0.885 | 0.747 | 0.888 | 0.797 | 0.883 |
| `name_token_intersection` | 0.773 | 0.787 | 0.723 | 0.820 | 0.699 | 0.823 | 0.744 | 0.816 |
| `name_token_jaccard` | 0.817 | 0.827 | 0.768 | 0.856 | 0.746 | 0.859 | 0.788 | 0.852 |
| `name_token_containment` | 0.811 | 0.828 | 0.763 | 0.855 | 0.737 | 0.857 | 0.787 | 0.853 |
| `name_length_ratio` | 0.726 | 0.709 | 0.713 | 0.719 | 0.727 | 0.725 | 0.701 | 0.714 |
| `name_script_mismatch` | 0.517 | 0.509 | 0.533 | 0.500 | 0.544 | 0.500 | 0.523 | 0.500 |
| `address_canonical_equal` | 0.565 | 0.522 | 0.540 | 0.545 | 0.559 | 0.569 | 0.522 | 0.523 |
| `address_char_similarity` | 0.961 | 0.930 | 0.924 | 0.956 | 0.964 | 0.959 | 0.891 | 0.955 |
| `address_token_set_similarity` | 0.985 | 0.979 | 0.980 | 0.983 | 0.985 | 0.986 | 0.977 | 0.983 |
| `address_token_intersection` | 0.911 | 0.884 | 0.917 | 0.929 | 0.937 | 0.943 | 0.902 | 0.918 |
| `address_token_jaccard` | 0.984 | 0.965 | 0.977 | 0.970 | 0.989 | 0.981 | 0.969 | 0.965 |
| `address_token_containment` | 0.983 | 0.967 | 0.985 | 0.968 | 0.989 | 0.981 | 0.982 | 0.962 |
| `address_length_ratio` | 0.704 | 0.648 | 0.701 | 0.647 | 0.767 | 0.659 | 0.643 | 0.642 |
| `query_number_count` | 0.479 | 0.475 | 0.465 | 0.489 | 0.468 | 0.490 | 0.462 | 0.488 |
| `target_number_count` | 0.422 | 0.416 | 0.396 | 0.434 | 0.398 | 0.436 | 0.394 | 0.430 |
| `shared_number_count` | 0.660 | 0.676 | 0.698 | 0.654 | 0.711 | 0.633 | 0.685 | 0.675 |
| `number_jaccard` | 0.736 | 0.762 | 0.851 | 0.694 | 0.867 | 0.663 | 0.835 | 0.723 |
| `query_numbers_in_target` | 0.690 | 0.712 | 0.772 | 0.660 | 0.791 | 0.632 | 0.755 | 0.686 |
| `target_numbers_in_query` | 0.733 | 0.763 | 0.794 | 0.717 | 0.795 | 0.691 | 0.792 | 0.743 |
| `number_set_equal` | 0.710 | 0.747 | 0.802 | 0.681 | 0.814 | 0.644 | 0.791 | 0.718 |
| `target_address_empty` | 0.512 | 0.511 | 0.511 | 0.512 | 0.511 | 0.513 | 0.511 | 0.512 |

Quantiles of key continuous features, positives (+) against negatives (−):

| Feature | Group | + p5 | + p25 | + p50 | + p75 | + p95 | − p5 | − p25 | − p50 | − p75 | − p95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `blocking_rank` | India | 1.000 | 1.000 | 2.000 | 4.000 | 14.000 | 4.000 | 13.000 | 24.000 | 36.000 | 48.000 |
| `blocking_rank` | US | 1.000 | 1.000 | 2.000 | 3.000 | 17.000 | 4.000 | 13.000 | 24.000 | 36.000 | 47.000 |
| `blocking_aggregate_rarity` | India | 0.200 | 1.208 | 2.785 | 5.048 | 9.530 | 0.025 | 0.050 | 0.083 | 0.167 | 0.522 |
| `blocking_aggregate_rarity` | US | 0.458 | 1.661 | 3.192 | 5.361 | 9.716 | 0.025 | 0.048 | 0.083 | 0.167 | 0.500 |
| `name_char_similarity` | India | 0.092 | 0.653 | 0.840 | 0.945 | 1.000 | 0.095 | 0.349 | 0.500 | 0.647 | 0.901 |
| `name_char_similarity` | US | 0.588 | 0.792 | 0.905 | 1.000 | 1.000 | 0.238 | 0.341 | 0.452 | 0.615 | 1.000 |
| `name_token_set_similarity` | India | 0.092 | 0.824 | 0.969 | 1.000 | 1.000 | 0.095 | 0.359 | 0.564 | 0.762 | 1.000 |
| `name_token_set_similarity` | US | 0.643 | 0.938 | 1.000 | 1.000 | 1.000 | 0.240 | 0.356 | 0.524 | 0.741 | 1.000 |
| `name_token_jaccard` | India | 0.000 | 0.400 | 0.600 | 1.000 | 1.000 | 0.000 | 0.000 | 0.167 | 0.333 | 0.800 |
| `name_token_jaccard` | US | 0.000 | 0.500 | 0.750 | 1.000 | 1.000 | 0.000 | 0.000 | 0.167 | 0.333 | 1.000 |
| `address_char_similarity` | India | 0.405 | 0.686 | 0.837 | 0.926 | 1.000 | 0.288 | 0.354 | 0.400 | 0.475 | 0.656 |
| `address_char_similarity` | US | 0.493 | 0.716 | 0.845 | 0.923 | 1.000 | 0.282 | 0.347 | 0.409 | 0.491 | 0.647 |
| `address_token_set_similarity` | India | 0.781 | 0.910 | 0.962 | 1.000 | 1.000 | 0.316 | 0.394 | 0.459 | 0.564 | 0.818 |
| `address_token_set_similarity` | US | 0.718 | 0.852 | 0.930 | 0.966 | 1.000 | 0.303 | 0.377 | 0.452 | 0.537 | 0.720 |
| `address_token_jaccard` | India | 0.267 | 0.615 | 0.778 | 0.889 | 1.000 | 0.000 | 0.045 | 0.095 | 0.167 | 0.375 |
| `address_token_jaccard` | US | 0.250 | 0.429 | 0.571 | 0.714 | 1.000 | 0.000 | 0.000 | 0.100 | 0.200 | 0.333 |
| `number_jaccard` | India | 0.000 | 0.667 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.250 | 0.500 | 1.000 |
| `number_jaccard` | US | 0.000 | 0.500 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.333 | 1.000 | 1.000 |

Rates (means of indicators) that explain the India/US contrast:

| Feature | India + | India − | US + | US − |
| --- | ---: | ---: | ---: | ---: |
| `name_canonical_equal` | 0.172 | 0.038 | 0.266 | 0.073 |
| `name_folded_equal` | 0.202 | 0.044 | 0.314 | 0.083 |
| `name_script_mismatch` | 0.146 | 0.080 | 0.000 | 0.000 |
| `address_canonical_equal` | 0.080 | 0.001 | 0.091 | 0.001 |
| `target_address_empty` | 0.032 | 0.010 | 0.036 | 0.012 |
| `number_set_equal` | 0.702 | 0.097 | 0.669 | 0.306 |
| `query_number_count` | 1.559 | 1.685 | 1.184 | 1.208 |
| `blocking_support_families` | 3.413 | 1.085 | 3.727 | 1.056 |
| `channel_name_x_address_word` | 0.659 | 0.093 | 0.830 | 0.106 |
| `channel_address_rare_word` | 0.366 | 0.061 | 0.126 | 0.020 |

## Observations

**Most informative families.**
- **Blocking evidence:** aggregate rarity has AUC 0.975, support families 0.927,
  and rank 0.062 (0.938 inverted).
- **Address token overlap:** token-set similarity 0.981, containment 0.973, Jaccard
  0.972, character similarity 0.944.
- **Name similarity:** token-set 0.840, character 0.836.
- **Numbers** are moderate overall (0.73–0.75).

Exact-equality indicators have low AUC (0.54–0.61) because most positives are not
exact. When they do fire, however, they are strongly enriched:
- folded name equal for 27.0% of positives against 6.7% of negatives;
- address equal for 8.7% against 0.1%.

**S2 against S3.**
- Address character similarity is much weaker for S3: 0.930 overall and 0.891 for
  India, against 0.961 for S2. This matches Phase 2: S3 reorders components and
  expands state codes.
- Order-insensitive token-set and containment features stay strong for S3
  (0.979 and 0.967).
- Exact address equality separates S2 more (0.565 against 0.522).
- S3 does slightly better on names (token-set 0.849 against 0.831) and on numbers
  (Jaccard 0.762 against 0.736).

**India against US.** India's harder recall recurs in the features:
- **Names are much weaker in India:** character similarity 0.764 against 0.882,
  Jaccard 0.768 against 0.856. 14.6% of Indian positives have a script mismatch
  (for example a Devanagari or Tamil target name against an ASCII S1 name),
  against 8.0% of Indian negatives. The US has none. For those pairs, name
  features are structurally uninformative.
- **Blocking evidence is weaker too:** aggregate rarity 0.962 against 0.983,
  support families 0.900 against 0.944.
- **Numbers are much stronger in India:** number Jaccard 0.851 against 0.694,
  number set equality 0.802 against 0.681. In India, 70.2% of positives but only
  9.7% of negatives have identical number sets. In the US the negatives agree far
  more often (30.6%), because short US addresses often carry a single house number.
- **Address token containment is at least as strong in India:** 0.985 against
  0.968.

The India matcher therefore needs to rely on address and numeric agreement,
treating a script mismatch as "name evidence unavailable" rather than as
disagreement. No country-specific feature is needed for this: the script and
number features are country-agnostic.

## Performance

| Measure | Value |
| --- | --- |
| Candidate pairs processed | 3,454,436 |
| Setup (manifest/hash checks, stores, offsets, truth load) | 28.1 s |
| Streaming wall time (S2 49.6 s + S3 49.8 s) | 99.4 s → **34,753 pairs/s** |
| Feature CPU in workers | 301.1 s → 11,473 pairs per CPU-second |
| Main-process blocking + text reads | 63.5 s |
| Total runtime, including diagnostics | 199.6 s |
| Peak RSS, main / largest worker | 4,982 MiB / 715 MiB |
| Sample matrix held in memory for diagnostics | 553.5 MiB |
| Persistent disk added | 95.6 MiB of text offsets |

The first complete run, before the five columns were removed, measured 137.3 s of
streaming and 269.0 s of feature CPU for the same pairs. Timings vary by roughly
±25% between runs with page-cache state.

The main-process RSS includes memory-mapped raw-file, store and index pages. The
one-time offset verification scans every raw file, so those file-backed pages are
reclaimable; anonymous memory is dominated by the diagnostic matrix.

**Where the CPU goes.** A single-batch profile (86,957 S2 pairs, 85,450 unique
targets) splits worker CPU as follows:

| Component | Share of worker CPU |
| --- | ---: |
| Record views (Phase 2 canonicalization) | 84.8% |
| Address token-set ratio | 4.9% |
| Token overlaps | 4.7% |
| Name token-set ratio | 2.6% |
| Number features | 1.4% |
| Name and address char ratios | 0.8–0.9% each |

No feature was disproportionately expensive, so none was removed for cost.

**Projected full development cost** (not executed). For 170,895,201 candidates:
- about **1.4 h wall** at the measured streaming rate with six workers;
- about **4.1 CPU-hours** of feature computation;
- **26.7 GiB** if the float32 matrix were materialized. It should not be; stream
  it into the consumer instead.

## Limitations

- **Univariate analysis only.** Correlations beyond the obvious duplicates are not
  quantified, and the AUCs are relative to the Phase 3B candidate distribution.
- **Uncalibrated evidence.** `blocking_rank` and the rarity features depend on the
  frozen ranking, and none of them is calibrated.
- **France cannot be measured.** It is absent from training. Accented-Latin
  handling (canonical versus folded equality) is tested, not measured.
- **Script is coarse.** The script feature is binary (any non-Latin letter). It
  captures the dominant Indic case but not transliteration quality.
- **Serial main thread.** About 64% of streaming time is serial blocking and text
  reads, which limits scaling beyond six workers.
- **Repeated canonicalization.** Record views are recomputed per batch. Targets
  rarely repeat within a batch (98% unique), but they will repeat across batches
  in a full-development pass.

## Recommendations for Phase 5 (deterministic matcher)

1. **Build rules from the strongest families.** Start with blocking aggregate
   rarity and support, address token-set/containment, name token-set and character
   similarity, and number agreement. Reserve exact name/address equality as
   high-precision shortcuts.
2. **Make name evidence optional when scripts differ.** When
   `name_script_mismatch` is set, fall back to address and number agreement,
   especially for India.
3. **Handle S3 addresses differently.** Use order-insensitive address features
   (token-set, containment) rather than character similarity for S3.
4. **Plan for singletons.** Every query, including the 5.6% true singletons, keeps
   up to 100 candidates. The matcher must be able to return an empty set, and
   thresholds must be tuned for Macro F0.5 on development and confirmed once on the
   frozen validation split.
5. **Consider per-query relative evidence.** Examples: the best score, or the
   margin between the best and second candidate. These are a Phase 5/6 design
   choice and were not built here.
6. **Stream at scale.** For full-development or supervised stages, stream
   `iter_features` batches directly into the consumer, or down-sample negatives,
   instead of writing the 26.7 GiB matrix.
