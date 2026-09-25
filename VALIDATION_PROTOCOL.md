# Frozen validation protocol — Phase 1.5

This stage adds evaluation and partition infrastructure only. It does not introduce
normalization, blocking, matching, model fitting, thresholds, or test predictions.
The official raw and resource files remain unchanged.

## Configuration and assignment

The committed `experiments/configs/validation.json` uses JSON to avoid adding a YAML
parser dependency. It contains seed **42**, validation fraction **0.1**, version
**source1-sha256-v1**, and the frozen expected fingerprint.

For an ID, form this UTF-8 byte string with literal NUL separators:

```text
split_version + NUL + decimal_seed + NUL + domain + NUL + entity_id
```

Take SHA-256, interpret all 256 bits as an unsigned big-endian integer, and assign
validation iff it is less than `floor(validation_fraction × 2^256)`. The threshold
uses exact rational arithmetic. Otherwise assign development. Source 1 uses the
domain `source1`; unlinked S2/S3 distractors use `distractor` and their full source
prefixed ID. Input row order, Python's randomized `hash()`, business strings, and
process random state do not enter the assignment.

Every labelled S2/S3 target inherits its owner's Source 1 partition. A target with
multiple owners, repeated input IDs, incomplete truth, or unknown positive targets
causes failure. The builder streams only the supplied training files, retains ID,
country and label-count scratch tables in SQLite, and assigns every unlinked target
independently using the distractor domain. No raw rows are copied to the split.
Training comprises India and US only; there are no invented French training records.

The final artifacts under ignored `data/interim/validation/` are six sorted,
ID-only TSV manifests (`development_source1.tsv` through `validation_source3.tsv`)
and `metadata.json`. Each manifest has the sole header `entity_id`. Sorting uses
SQLite binary text order, not numeric suffix order. Temporary databases are removed.

## Freeze and reproducibility

Metadata records the exact raw-file SHA-256 hashes, manifest SHA-256 hashes,
configuration, integrity checks, partition distributions and representativeness
checks. The fingerprint is SHA-256 of the metadata excluding its own fingerprint
field, serialized as sorted-key JSON with separators `(',', ':')`. It contains no
absolute paths or timestamps. The expected fingerprint is committed in the config.

Preparation verifies existing artifacts and raw hashes before reusing them.
`--rebuild` recreates all memberships and requires the result to equal the existing
metadata and committed fingerprint. It cannot silently replace a changed split.
The evaluator verifies the config, metadata, all six manifests and the ground-truth
hash before scoring. Moving a checkout does not change membership or fingerprints.
Reordering raw rows preserves membership and sorted manifest bytes, but deliberately
changes the raw input fingerprint; a frozen run therefore detects that alteration.

To change the protocol, create an explicitly reviewed new version and output
directory, regenerate, inspect distributions/leakage checks, and update its expected
fingerprint and documentation. Do not edit the frozen seed/fraction or remove its
fingerprint to make an experiment pass. Experiments must record the fingerprint.

## Coverage, leakage and representativeness

Unique ID indexes and a single partition value enforce one assignment per record.
The builder additionally checks Source 1 truth coverage, missing/unknown targets,
positive target/owner partition agreement, per-source partition overlap and exact
target coverage/counts. A failure prevents publication of the manifests.

Development and validation metadata include country counts/fractions, singleton,
one-match and multi-match fractions, the full match-count histogram, mean matches,
S2/S3 positive link counts and S2/S3 distractor counts. For each country and each
match-count bin, material imbalance is defined before inspection as an absolute
fraction gap exceeding both 0.5 percentage points and five pooled binomial standard
errors. Mean-match imbalance must exceed both 2% of the pooled mean and five pooled
standard errors. The hash split must pass these checks; otherwise preparation
stops for a reviewed deterministic stratified version instead of publishing an
unbalanced split. Rare bins are assessed in absolute, not relative, percentage terms.

## Frozen real-data split

Fingerprint: `1dd67d77fee74aa0b64477e10f09377ac22854f5eb13780f3aacc4baf94e3645`.

| Statistic | Development | Validation |
| --- | ---: | ---: |
| Source 1 entities | 1,986,411 | 220,410 |
| Share of Source 1 | 90.0123% | 9.9877% |
| India | 795,204 (40.0322%) | 87,984 (39.9183%) |
| US | 1,191,207 (59.9678%) | 132,426 (60.0817%) |
| Singleton percentage | 5.5823% | 5.6077% |
| Exactly one match | 5.3938% | 5.4503% |
| Multiple matches | 89.0239% | 88.9420% |
| Mean matches | 3.461908 | 3.455342 |
| Source 2 target pool | 4,531,619 | 502,997 |
| Source 2 positive links | 3,325,308 | 368,311 |
| Source 2 distractors | 1,206,311 | 134,686 |
| Source 3 target pool | 4,757,870 | 527,733 |
| Source 3 positive links | 3,551,465 | 393,281 |
| Source 3 distractors | 1,206,405 | 134,452 |

| Matches | Development entities | Validation entities |
| ---: | ---: | ---: |
| 0 | 110,887 | 12,360 |
| 1 | 107,144 | 12,013 |
| 2 | 337,398 | 37,814 |
| 3 | 478,248 | 52,593 |
| 4 | 435,641 | 48,474 |
| 5 | 289,923 | 32,034 |
| 6 | 148,352 | 16,516 |
| 7 | 57,649 | 6,319 |
| 8 | 16,830 | 1,850 |
| 9 | 3,809 | 396 |
| 10 | 496 | 38 |
| 11 | 34 | 3 |

The largest country gap is 0.1139 percentage points; the largest match-count-bin gap is 0.2145 percentage points. All predefined balance checks pass, so no stratification or seed search was used.

All 12 recorded integrity check counts are zero: missing/unknown Source 1 labels, positive-owner partition mismatches, each source overlap, unknown positive targets, unassigned targets, and target count differences. Source 1 coverage totals 2,206,821; Source 2 pools total 5,034,616; Source 3 pools total 5,285,603. Labelled links total 7,638,365. Every positive follows its owner and each target appears in one pool.

Real-data file-evaluator checks on all 220,410 validation entities:

| Fixture | Macro F0.5 | TP | FP | FN | Exact-match entities |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation truth copied as predictions | 1.0 | 761,592 | 0 | 0 | 220,410 |
| Every prediction empty | 0.05607731046685722 | 0 | 0 | 761,592 | 12,360 |
| Truth copy plus one development-owned target | 0.999998703714506 | 761,592 | 1 | 0 | 220,409 |

The second score is exactly `12,360 / 220,410`, the validation singleton fraction.
The rows above were rerun under evaluation protocol `full-training-search-universe-v2`
with the full search universe (5,034,616 Source 2 and 5,285,603 Source 3 IDs). The
truth-copy and all-empty scores are identical to the original validation-pool run.
In the third fixture, `S2-681193310` is a labelled target of development entity
`S1-965667`. Added to validation entity `S1-29845983`, whose truth has two targets,
it was accepted and scored as one FP; the entity score is `10 / 14`. A fixture
predicting `S2-does-not-exist` was rejected.
The full diagnostics are under ignored `data/interim/validation_sanity.json`.
All 71 tests pass, covering the scorer, parser rejection paths, split reproduction,
reordered input, target ownership and coverage, distractors, fingerprint tampering,
the Phase 1 hash workflow, cross-partition false positives, nonexistent-target
rejection, isolated training pools and the frozen real-data split and universe counts. The raw/official hash-only verification passed
without rerunning the Phase 1 statistical audit.

A second full real-data build with `--rebuild` reproduced all six manifest hashes,
all metadata and the committed fingerprint exactly. All temporary rebuild tables
were removed. A final hash check confirmed all seven raw TSVs and all three
official resources are unchanged.

## Training pool versus validation search universe

The target ownership partition and the evaluation search universe are separate
concepts:

```text
Development training pool:
isolated development-labelled target pool

Validation retrieval/search universe:
complete training Source 2 + Source 3 corpus
```

The **development training pool** is `development_source2.tsv` and
`development_source3.tsv`: development-owned positives plus development-hashed
distractors. It never contains a target labelled for a validation Source 1 entity,
so models trained on it cannot see validation positives. `training_target_pool()`
in `src/data/validation.py` returns it. Ownership assignment, both pools and the
leakage checks above are unchanged.

The **validation search universe** is every training Source 2 and Source 3 record,
as in the real challenge, where each Source 1 entity searches the complete Source 2
and Source 3 databases. `search_universe()` returns it as the union of the
development and validation manifests for each source. That union is exact: the
frozen integrity checks show zero unassigned records, zero count differences and
zero overlap, and a direct comparison with the raw ID columns found them
byte-identical after sorting. It contains **5,034,616 Source 2** and
**5,285,603 Source 3** records. Using the ID-only manifests means the evaluator
never parses raw rows, names or addresses.

A validation prediction may reference any target in this universe. A target that
is not in the entity's truth set, including one owned by or labelled for a
development entity, is accepted and counted as a false positive. Only IDs outside
the universe (nonexistent records) are rejected.

The evaluation protocol is versioned separately from the split as
`full-training-search-universe-v2` (`EVALUATION_PROTOCOL`). It is reported in every
evaluator output. The Source 1 split, its configuration, manifests and fingerprint
are unchanged by this evaluator correction. Before this correction, validation
predictions were restricted to the validation-owned pools, a
non-competition-faithful search space. Scores produced before the correction are
not comparable with scores after it whenever a prediction touched a
development-owned target.

## Exact evaluator contract

For each selected Source 1 ID, truth and prediction are sets. With nonempty truth:

```text
F0.5_i = 5 TP / (5 TP + 4 FP + FN)
```

This is exactly the specified `1.25 TP / (1.25 TP + FP + 0.25 FN)`. Empty truth and
empty prediction score 1; empty truth with any predicted target scores 0; nonempty
truth with no predicted targets scores 0. The challenge score is the arithmetic
mean over every selected Source 1 entity, computed with `math.fsum` for stable
accumulation. An empty evaluation subset is rejected.

Prediction input uses the exact official `source1_entity_id<TAB>matched_entity_ids`
header, two fields per row and unquoted comma-separated lists. CRLF or LF line
endings are accepted. Every selected entity must appear once, including explicit
empty lists. Missing rows are errors, not implicit singleton predictions. Extra
Source 1 IDs, duplicate rows, duplicate targets in a row, wrong prefixes, malformed
fields and targets outside the selected target universe are rejected. Using
the same valid target in different prediction rows is permitted and scored against
each truth set; there is no extra one-to-one constraint.

`evaluate_predictions` takes an explicit `target_universe` mapping from Source 2 and
Source 3 to the ID manifests a prediction may reference. Both selected truth targets
and predicted targets must belong to it, and repeated IDs across its manifests are
rejected. It returns `target_universe_counts`. The executable validates the
manifests' frozen provenance before calling it. Temporary
SQLite lookup tables bound memory regardless of file ordering. Business-name and
address source columns are never loaded by the evaluator.

Output includes Macro F0.5, evaluated entities, exact-match entities, true
singletons, correct singletons, singleton false merges, TP/FP/FN, counts of
zero/one/multiple predictions, and explicitly named
`micro_precision_diagnostic` / `micro_recall_diagnostic`. These two diagnostics use
zero when their denominator is zero and are **not** the leaderboard score.
Blocking recall and a decision threshold are unavailable at this stage and are
not invented. Every CLI result includes the frozen split fingerprint.

## Commands

With the project environment active:

```bash
python scripts/prepare_validation_split.py
python scripts/prepare_validation_split.py --rebuild
python scripts/evaluate_model.py --predictions data/interim/experiment_predictions.tsv \
  --partition validation --output data/interim/experiment_metrics.json
python scripts/evaluate_model.py --predictions data/interim/dev_predictions.tsv \
  --partition development --target-universe partition-pool
python scripts/evaluate_model.py --sanity-checks \
  --output data/interim/validation_sanity.json
python scripts/inspect_data.py --hashes-only
python -m pytest -q
```

`--partition development` selects development Source 1 entities. `--target-universe`
defaults to `full`, the complete training corpus, for both partitions.
`--target-universe partition-pool` restricts the targets to the isolated development
training pool. It is a training diagnostic, is accepted only with
`--partition development`, and cannot be used for validation or sanity checks.
Reports record `evaluation_protocol` and `target_universe`. No prediction argument is
silently synthesized; evaluating without a prediction file or `--sanity-checks`
prints usage and exits. `--sanity-checks` explicitly creates temporary validation fixtures,
exercises the same file evaluator against the full search universe, and deletes
them: a truth copy, all empty, a truth copy plus one development-labelled target
on the first non-singleton validation entity (must be accepted and add exactly one
FP), and a truth copy plus `S2-does-not-exist` (must be rejected). They are evaluator tests, not model or test-set predictions. No final
submission files are generated.

## Limits

The organizers' executable scorer and public/private test split remain unavailable.
The local score follows the exact formula and empty-set conventions explicitly
requested for Phase 1.5. Remote handling of nonexistent IDs remains contradictory
in the official README and validator; local evaluation rejects IDs outside the full
training corpus. ID/link
partition isolation does not prove absence of unlabeled semantic business overlap
among distractors or repeated strings. No model performance is claimed by the
sanity checks, and the validation distribution cannot establish performance on
France, which is absent from training.
