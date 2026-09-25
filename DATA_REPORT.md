# Phase 1 — Data Understanding

This report audits only the supplied challenge package. No external business data,
normalization, candidate generation, matching features, models, or predictions were
used. Raw and official files are preserved. Findings below distinguish measured
properties from the rules in the official resources.

## Reproduction and scope

```bash
.venv/bin/python scripts/inspect_data.py --inventory-only
.venv/bin/python scripts/inspect_data.py
.venv/bin/python -m pytest -q
```

The full inspection writes `data/interim/data_profile.json`; it does not print raw
tables. Sources and ground truth are streamed in batches of 20,000 records into
scratch SQLite tables under `data/interim/`. Exact duplicate and reference checks
operate on those tables. No full pandas DataFrame is required, and source files
are parsed once per inspection run. Scratch databases are removed on success or
normal exception exit; several GB of free scratch disk are needed. JSON contains
`complete: true` only after all checks finish. Character-length percentiles use the
nearest-rank definition, and duplicate counts mean excess rows after the first
occurrence. Missing means an empty or whitespace-only field; literal `NA`, `NULL`,
and similar strings are counted separately and are never silently converted.

Initial headers and tiny samples were checked before the full audit. Tests read
only ten rows from each real file, alongside small synthetic fixtures; the full
inspection, rather than the sample tests, establishes whole-file integrity.
SHA-256 fingerprints are recorded in `data/interim/phase1_input_hashes.json` for all
seven raw TSVs and all three official resources.

## Schema and loading contract

All six source files have exactly these ordered columns:

| Column | Logical / loader type | Meaning |
| --- | --- | --- |
| `entity_id` | string | Record ID, prefixed `S1-`, `S2-`, or `S3-` according to source |
| `business_name` | string | Unstructured business/trade name |
| `business_address` | string | Unstructured address; locality, region, postal code and landmarks may be embedded |
| `country` | string | Open country label; not a closed training-only enumeration |

There are no separate phone, website, postal-code, locality, latitude/longitude,
or source columns. Some name strings resemble domains, but no dedicated URL field
exists. ID suffixes are opaque identifiers; their numbers must not be treated as
matching evidence or numeric measurements. Source provenance comes from the file
and ID prefix. Matching IDs from Source 2 and Source 3 are distinguishable directly
by their prefix, even when interleaved in the same list.

The ground truth has exactly two ordered string columns:
`source1_entity_id`, `matched_entity_ids`. It is one row per reference entity, not
one row per true link. A trailing tab followed by a newline represents an empty
match list. A single target is a plain `S2-...` or `S3-...` string. Multiple targets
are comma-separated, potentially with several records from either source.

`src/data/load.py` now provides validated, streaming `iter_source` and
`iter_ground_truth` loaders. The latter returns `(source1_id, tuple_of_targets)`,
with `()` for a singleton and original target order preserved. It rejects wrong
headers, wrong field counts, wrong source prefixes, malformed list items and
repeated IDs within a list. It does not accumulate millions of IDs to check
uniqueness: whole-file duplication and referential integrity belong to the disk
backed inspector. `load_source` / `load_ground_truth` are convenience pandas loaders
for suitably sized inputs; all columns explicitly use pandas string dtype. Empty
strings and Unicode are preserved, and countries remain unrestricted. The generic
`read_table` also disables implicit numeric and NA conversion.

## Measured file inventory and row counts

All counts are exact, excluding headers. Total raw size is 2,520,573,701 bytes (2.521 GB decimal; 2.347 GiB). There are 24,229,173 source records plus 2,206,821 truth rows.

| File | Bytes | MiB | Data rows |
| --- | ---: | ---: | ---: |
| `data/raw/train/train_source1.tsv` | 210,069,713 | 200.34 | 2,206,821 |
| `data/raw/train/train_source2.tsv` | 489,301,488 | 466.63 | 5,034,616 |
| `data/raw/train/train_source3.tsv` | 503,705,637 | 480.37 | 5,285,603 |
| `data/raw/test/test_source1.tsv` | 175,022,086 | 166.91 | 1,732,544 |
| `data/raw/test/test_source2.tsv` | 509,456,422 | 485.86 | 4,887,273 |
| `data/raw/test/test_source3.tsv` | 506,002,772 | 482.56 | 5,082,316 |
| `data/raw/train/train_ground_truth.tsv` | 127,015,583 | 121.13 | 2,206,821 |

Official resources discovered (no additional sample submission or scoring code):

| File | Bytes |
| --- | ---: |
| `official/Documentation_template.md` | 2,175 |
| `official/README.md` | 14,130 |
| `official/utils/validate_submission.py` | 13,687 |

### Country distributions

| Source file | India | US | France |
| --- | ---: | ---: | ---: |
| train_source1 | 883,188 (40.02%) | 1,323,633 (59.98%) | 0 (0.00%) |
| train_source2 | 2,017,799 (40.08%) | 3,016,817 (59.92%) | 0 (0.00%) |
| train_source3 | 2,115,547 (40.02%) | 3,170,056 (59.98%) | 0 (0.00%) |
| test_source1 | 809,986 (46.75%) | 663,106 (38.27%) | 259,452 (14.98%) |
| test_source2 | 2,312,565 (47.32%) | 1,871,330 (38.29%) | 703,378 (14.39%) |
| test_source3 | 2,405,000 (47.32%) | 1,945,701 (38.28%) | 731,615 (14.40%) |

France is absent from training and present in all three test sources. Its 259,452 Source 1 entities must all be represented later. Country proportions also shift between splits.

### Missingness and literal null-like strings

Every source column is read as text. IDs, names and countries have **0 missing values (0%)** in all six files. Addresses are the only source field with empty values. No whitespace-only fields or leading/trailing whitespace were found in any source column. The following null-like name counts use case-insensitive literal membership in `NA`, `N/A`, `NaN`, `NULL`, `None`, `<NA>`; whether these names are intended missing-value sentinels is not documented.

| Source file | Empty addresses | Address missing rate | Null-like literal names |
| --- | ---: | ---: | ---: |
| train_source1 | 0 | 0.0000% | 0 |
| train_source2 | 168,967 | 3.3561% | 6 |
| train_source3 | 175,916 | 3.3282% | 18 |
| test_source1 | 0 | 0.0000% | 0 |
| test_source2 | 129,408 | 2.6479% | 49 |
| test_source3 | 136,098 | 2.6779% | 61 |

All other source-field null-like literal counts are zero. Empty addresses remain empty strings; default pandas NA inference would change the meaning of some name values.

### Exact duplicates and ID uniqueness

All IDs have the prefix required by their source file. ID uniqueness equals the row count in every source. There are **zero duplicate IDs and zero duplicate complete rows** in all six files. Repeated profiles below exclude ID and compare `(business_name, business_address, country)` without any cleaning; repeated names/addresses are also raw exact comparisons.

| Source file | Unique names | Excess repeated names | Unique addresses | Excess repeated addresses | Excess repeated profiles |
| --- | ---: | ---: | ---: | ---: | ---: |
| train_source1 | 1,539,229 | 667,592 | 2,130,606 | 76,215 | 0 |
| train_source2 | 4,402,009 | 632,607 | 4,337,262 | 697,354 | 25,873 |
| train_source3 | 4,651,609 | 633,994 | 4,632,765 | 652,838 | 18,860 |
| test_source1 | 1,238,867 | 493,677 | 1,677,483 | 55,061 | 0 |
| test_source2 | 4,311,041 | 576,232 | 4,224,784 | 662,489 | 22,641 |
| test_source3 | 4,521,929 | 560,387 | 4,456,436 | 625,880 | 16,293 |

Address distinct/duplicate counts include the empty string. Source 1 has no repeated full profiles, but separate names and addresses recur. For example, training Source 1 contains `Primary Care Group` 253 times, `Ear Nose & Throat Group` 251 times and `Pediatric Group` 222 times. Its most frequent exact address appears 14 times. Identical names or addresses alone are therefore unsafe as universal match assertions. Different target IDs with identical profiles are still different challenge records and must remain available for multi-match labels.

### Field lengths and character distributions

Lengths are Unicode code points, including punctuation/spaces, and include empty strings. ID lengths vary; suffixes are not uniformly zero-padded.

| Source file | ID min–max | Name min / median / p95 / max (mean) | Address min / median / p95 / max (mean) |
| --- | ---: | --- | --- |
| train_source1 | 6–12 | 3 / 24 / 37 / 105 (24.03) | 11 / 41 / 103 / 256 (52.07) |
| train_source2 | 6–12 | 2 / 25 / 40 / 104 (25.10) | 0 / 37 / 96 / 249 (46.23) |
| train_source3 | 5–12 | 2 / 25 / 42 / 123 (25.20) | 0 / 42 / 91 / 240 (46.71) |
| test_source1 | 6–12 | 3 / 24 / 36 / 92 (23.84) | 11 / 50 / 105 / 268 (57.21) |
| test_source2 | 6–12 | 2 / 25 / 42 / 102 (25.70) | 0 / 43 / 99 / 269 (50.41) |
| test_source3 | 4–12 | 2 / 25 / 42 / 103 (25.66) | 0 / 43 / 94 / 267 (48.74) |

| Source file | Non-ASCII names | Devanagari names | Non-ASCII addresses | Devanagari addresses | Nonempty addresses without a digit |
| --- | ---: | ---: | ---: | ---: | ---: |
| train_source1 | 0 (0.00%) | 0 | 554 (0.03%) | 0 | 77,037 |
| train_source2 | 764,608 (15.19%) | 269,424 | 478,453 (9.50%) | 277,523 | 301,974 |
| train_source3 | 606,737 (11.48%) | 158,003 | 476,588 (9.02%) | 275,861 | 310,390 |
| test_source1 | 40,789 (2.35%) | 0 | 73,800 (4.26%) | 0 | 71,838 |
| test_source2 | 928,158 (18.99%) | 309,103 | 720,665 (14.75%) | 318,369 | 228,622 |
| test_source3 | 737,515 (14.51%) | 181,068 | 729,222 (14.35%) | 319,073 | 245,395 |

Devanagari means any code point in U+0900–U+097F, a descriptive script check, not a language classifier. Training Source 1 names are entirely ASCII, while target sources include hundreds of thousands of non-ASCII names. Transliteration/script differences and French accents must be considered later; deleting all non-ASCII characters would discard useful evidence. Addresses without digits show that street-number or postal-code-dependent rules cannot cover all records. Nonempty addresses may still be incomplete; absence of an empty value is not a completeness guarantee. The official noise description also warns about typos, legal suffixes, abbreviations, trade names, reordered components and landmark references. These are future analysis needs, not transformations applied in this phase.

## Ground-truth statistics and integrity

There are 2,206,821 truth rows and 2,206,821 unique Source 1 IDs. No Source 1 ID is empty, duplicated, missing from truth, or unknown to train Source 1. There are no duplicate complete truth rows, no malformed ID lists, and no repeated target IDs within a list. Every target ID exists in its indicated training source. No target is labelled as belonging to more than one Source 1 entity. Train and test ID overlap is **zero for each of S1, S2 and S3**; this does not prove that underlying businesses never overlap across splits.

| Match category | Source 1 entities | Percentage of all train Source 1 |
| --- | ---: | ---: |
| Zero (singletons) | 123,247 | 5.5848% |
| Exactly one | 119,157 | 5.3995% |
| Multiple (2–11) | 1,964,417 | 89.0157% |

The raw `matched_entity_ids` column has 123,247 empty strings (5.5848%); each is an intentional no-match label, not an absent annotation. `source1_entity_id` has 0% missingness. Nonempty target lists have valid comma-separated IDs without separator whitespace. Match-list order is preserved for loading but has no set-based scoring meaning.

Actual small examples from the supplied truth:

| Case | Source 1 ID | `matched_entity_ids` |
| --- | --- | --- |
| zero | `S1-302869473` | (empty string) |
| one | `S1-116043204` | `S3-85523430` |
| multiple | `S1-965667` | `S2-681193310,S2-743505751,S3-775321672,S3-11291185,S3-860443364` |

### Complete match-count distribution

| Matches per entity | Entities | Percentage |
| ---: | ---: | ---: |
| 0 | 123,247 | 5.5848% |
| 1 | 119,157 | 5.3995% |
| 2 | 375,212 | 17.0024% |
| 3 | 530,841 | 24.0546% |
| 4 | 484,115 | 21.9372% |
| 5 | 321,957 | 14.5892% |
| 6 | 164,868 | 7.4708% |
| 7 | 63,968 | 2.8986% |
| 8 | 18,680 | 0.8465% |
| 9 | 4,205 | 0.1905% |
| 10 | 534 | 0.0242% |
| 11 | 37 | 0.0017% |

Mean = 3.461253; median = 3; p95 = 6; maximum = 11.

### Links by target source

| Target source | True links | Share of all links | Source 1 entities with ≥1 such link | Unique linked targets | Target records with no labelled S1 link |
| --- | ---: | ---: | ---: | ---: | ---: |
| Source 2 | 3,693,619 | 48.3561% | 1,919,076 (86.9611%) | 3,693,619 | 1,340,997 |
| Source 3 | 3,944,746 | 51.6439% | 1,940,545 (87.9340%) | 3,944,746 | 1,340,857 |

Total true links: **7,638,365**. The entity counts in the source columns overlap when an entity has targets in both sources. Each unique target occurs in one truth list in this supplied training set. The remaining target records are distractors relative to labelled Source 1 links; no claim is made about their mutual business identity.

| Target-source coverage | Source 1 entities | Percentage |
| --- | ---: | ---: |
| Neither | 123,247 | 5.5848% |
| Source 2 only | 143,029 | 6.4812% |
| Source 3 only | 164,498 | 7.4541% |
| Both | 1,776,047 | 80.4799% |

### Ground truth by reference country

| Country | Source 1 entities | Zero | One | Multiple | Links to S2 | Links to S3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| India | 883,188 | 49,351 (5.5878%) | 47,468 (5.3746%) | 786,369 (89.0376%) | 1,480,545 | 1,579,298 |
| US | 1,323,633 | 73,896 (5.5828%) | 71,689 (5.4161%) | 1,178,048 (89.0011%) | 2,213,074 | 2,365,448 |

### Raw-field agreement on supplied positive links

These are descriptive string-equality counts on labelled links, without normalization, fuzzy comparison or a matching model. All true links agree on country.

| Link source | Identical raw names | Identical raw addresses | Both name and address identical | Country mismatches |
| --- | ---: | ---: | ---: | ---: |
| S1 → S2 | 174,617 (4.7275%) | 5 (0.000135%) | 1 | 0 |
| S1 → S3 | 179,501 (4.5504%) | 170,069 (4.311279%) | 17 | 0 |

Source 2 and Source 3 show different raw address-agreement rates; source-specific analysis will matter later. Exact unprocessed names/addresses alone have a low recall ceiling on these positive labels. Multiple targets per source are common, so restricting each reference to its nearest single target would contradict the labels. The observed country consistency may help future blocking, but must not be assumed to prove test-country consistency. The singleton rate and the many unlinked target records make abstention and false-positive prevention relevant for later thresholding.

The unrestricted train comparison space is 22,774,876,013,799 pairs (22.77 trillion).
The unrestricted test comparison space is 17,272,751,604,416 pairs (17.27 trillion).

These sizes motivate later blocking-recall and candidate-volume measurement; no candidates were generated here.



## Official submission contract

Source: [official README](official/README.md), particularly **Output Format**,
**Constraints**, **Evaluation Criteria**, and **Final Submission Package**.

| Artifact | Exact ordered header | Row meaning |
| --- | --- | --- |
| `output/matching_results.tsv` | `source1_entity_id<TAB>matched_entity_ids` | Final target IDs for one test Source 1 entity |
| `output/candidate_pairs.tsv` | `source1_entity_id<TAB>candidate_entity_ids` | Final candidates actually passed to the matching model for that entity |

Both are UTF-8, tab-separated files with a header, exactly two columns, no pandas
index column, and exactly one row for **every** test Source 1 entity, including
France and entities with no selected IDs. The second column is an unquoted
comma-separated list with no added spaces, or an empty string (retain the tab).
Use LF newlines for compatibility with the literal line parser. No specific row
order or list order is stated; the validator uses sets. Deterministic ordering
will be sensible later but is not an official scoring requirement.

Source 1 IDs must occur once each and must be from test Source 1. Every listed
target must be an existing test Source 2 or Source 3 ID, with no duplicates within
one list and no Source 1 targets. The validator does **not** enforce uniqueness of
a target across different Source 1 rows. Final matches must be a subset of the
corresponding candidates. The candidate file represents the **last** filtering
stage before model inference, not a larger early blocking pool and not an edge
list with one row per pair.

Only `matching_results.tsv` is uploaded for leaderboard scoring. Both files are
required in the final zip, along with reproducible code/dependencies and the
methodology document. No sample submission file is supplied; examples are inline
in the README. No submission artifacts were created during this phase.

### What the supplied validator actually checks

Source: [unchanged validator](official/utils/validate_submission.py), especially
`validate_id_list_file`, `load_match_targets`, and `validate`.

The future command, adjusted to the repository layout, is:

```bash
python official/utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir data/raw/test \
  --check-ids
```

- Errors include missing matching file, empty/broken header, comma-delimited header,
  nonblank rows without a tab, repeated Source 1 rows, repeated targets within a
  list, Source 1 targets, wrong target prefixes, missing required Source 1 IDs,
  and extra Source 1 IDs. With `--check-ids`, nonexistent target IDs are errors too.
- ID existence checking is **off by default**. It loads both target ID sets only
  with `--check-ids`; if a target source file is missing, it warns and skips that
  check. A default `PASS` therefore does not establish target-ID validity.
- A missing candidate file only warns locally, even though the final zip requires
  it. Matches absent from candidate lists also only warn, never fail.
- Headers are stripped and lowercased before comparison, but canonical exact names
  should be written. Data IDs are not generally trimmed. The parser splits literal
  tabs/commas rather than using CSV quote decoding. Quoted ID lists are unsuitable.
- Blank lines are ignored. There is no independent exact-field-count validation
  after the first tab; extra tabs in the remaining string are not robustly rejected
  in every case when ID checking is off. Follow the documented two-column format
  even if some malformed cases happen to pass locally.
- Exit 0 means no validator errors, exit 1 means errors. It never reads ground
  truth and never computes a score. No scoring implementation is supplied.

There is an official inconsistency: the README says nonexistent IDs are rejected;
the validator's comments/help say the remote scorer merely lowers the score.
Locally, `--check-ids` does add these to errors. The safe format contract is to use
only real test IDs; remote rejection behavior cannot be established from these
conflicting resources.

## Exact documented scoring and limits of confirmation

For each Source 1 entity, compare its predicted target set with its true target
set. For nondegenerate sets, precision is `TP / (TP + FP)` and recall is
`TP / (TP + FN)`. The official formula is:

```text
F0.5_i = 1.25 * precision_i * recall_i / (0.25 * precision_i + recall_i)
Macro F0.5 = sum(F0.5_i) / number_of_Source_1_entities
```

The README explicitly says to average **per-entity F0.5**, across **all** reference
entities. It does not say to compute F0.5 from global link counts, nor to plug
macro-averaged precision and recall into the formula. An entity with many matches
has the same macro weight as an entity with one match or none.

Singleton behavior is explicit: if the true set is empty, an empty prediction
scores **1.0**, and any nonempty prediction scores **0.0**. For a nonempty true set
and an empty prediction, the standard algebraic continuation gives **0.0**:

```text
F0.5_i = 1.25 * TP / (1.25 * TP + FP + 0.25 * FN)
```

This expression is equivalent where precision/recall are defined and avoids the
undefined precision when no matches are predicted. The README does not explicitly
specify that denominator-zero precision convention; no scorer code is available
to inspect. The singleton override is documented, while the non-singleton empty
prediction case is an inferred standard convention to retain explicitly in later
evaluation tests. The supplied resources also do not specify separate aggregate
precision/recall reporting conventions. Those must be labelled when implemented.

The README's informal phrase “precision 2× over recall” should not replace the
formula: `beta=0.5`, hence `beta²=0.25`; in the count form FP has four times FN's
coefficient. The README example with two correct targets and one false target
scores `5/7 ≈ 0.714286`, consistent with the formula.

The public/private leaderboard split is applied by organizers to test predictions;
its membership is not provided. The README asks teams to hold out training data
for their own validation. No fixed official training validation split is supplied,
and none was created or changed here. No matching evaluation or improvement claim
was made during Phase 1.

## Documentation template and future work boundaries

[Documentation_template.md](official/Documentation_template.md) asks for team
metadata, an executive summary, problem analysis, solution strategy, candidate
keys/counts/recall safeguards, model/features/threshold selection, Macro F0.5 and
error analysis, conclusions, and reproducible code artifacts. It adds no scoring
rules or new dataset schema. It remains unchanged and unfilled at this stage.

Future work must preserve country and Unicode information, quantify blocking
recall, permit multiple targets from either source, include singleton decisions,
and tune to the documented per-entity metric. The data observations do not justify
any matching rule or threshold without later validation. Existing normalization
and evaluation modules were not modified or invoked as a matching pipeline.

## Remaining assumptions and unresolved questions

- The remote scoring implementation is unavailable; empty-prediction precision and
  separate aggregate precision/recall conventions are not explicitly documented.
- The README and validator disagree on remote rejection of nonexistent target IDs.
- No test truth, public/private membership, or official fixed training holdout is
  supplied. Training statistics cannot establish test accuracy or the France
  singleton/match distribution.
- Country consistency observed in training is evidence about this dataset, not an
  official guarantee that a future country restriction preserves every test match.
- Repeated business strings do not by themselves establish either entity equality
  or data errors. Unlinked target records have no labelled Source 1 link in training;
  their underlying relationship to each other is not specified.

## Verification

- Full streaming audit completed successfully for all seven TSVs; the complete JSON
  is in `data/interim/data_profile.json` and scratch databases were removed.
- SHA-256 before/after comparison passed for all seven raw files and all three
  official resources. No raw or official contents changed.
- All 32 tests passed under Python 3.14.7, pandas 3.0.6 and pytest 9.1.1.
- Tests cover literal-string/Unicode preservation, unseen countries, explicit
  singleton labels, mixed/multiple targets, malformed schemas/IDs, lazy reads,
  exact duplicates, reference errors, deterministic inspection, and small samples
  from every real input file.
- Changes are confined to `src/data/`, `scripts/inspect_data.py`, `tests/test_data.py`,
  this report, and ignored derived inspection files under `data/interim/`.
  No validation split, experiment parameters, matching code or output predictions
  were changed or produced.

