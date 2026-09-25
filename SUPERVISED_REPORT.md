# Phase 6 — Supervised matching over the frozen Phase 5 candidate screen

## Scope and protocol

This experiment asks whether a compact LightGBM classifier over the existing 42
Phase 4 features materially improves on the deterministic Phase 5 matcher.
The starting checkpoint is `99fe7b90bfff4e79f90b53d482e42b8ca013734a` on
`phase6-supervised`. Candidate generation, ranking, normalization, feature
semantics, the Phase 5 prefilter, and the frozen validation split are unchanged.
No external challenge data, test inference, submission, or Phase 7 optimization
is performed. Existing Phase 5 submission files are preserved.

The only shared-pipeline change is an optional screened-feature consumer in
`collect_candidates`. Existing calls have identical behavior. Development copies
only screened feature rows into a preallocated RAM matrix. Validation, if admitted
by the gate, scores screened batches immediately and retains compact candidate
metadata, rule evidence, labels and probabilities; it never retains a complete
validation feature matrix. Both paths reuse the existing candidate generation,
feature extraction, labeling, prefilter and canonical evaluator.
Inference uses the saved binary Booster's positive-class probabilities, equivalent
to the fitted classifier's `predict_proba(...)[:, 1]`; a focused test checks this.

The feature extraction batches contain 1,000 queries and use three workers to
limit transient memory. Models use six CPU threads. The RAM matrix contains only
the 40,170-query development sample, never full development. Disk and available
system memory are checked at each consumed batch and before major stages. The
free-disk guard is 10.5 GiB; the available-memory guard is 1,024 MiB. Core dumps
are disabled by the runner.

## Fixed experiment design

Config: `experiments/configs/supervised_v1.json`.

- Existing development sample: salt `blocking-sample-v1`, fraction 0.02.
- One internal query split: SHA-256 through the existing `in_sample` helper,
  salt `supervised-query-split-v1`, tune fraction 0.20. Every pair inherits its
  Source 1 query's partition. Country is not used to form the split.
- Train on all screened positives and all screened negatives in the train
  partition. No negative sampling, weighting, or extra feature family.
- Two configurations only: 31 or 63 leaves, learning rate 0.05, at most 500
  rounds, minimum 100 rows per leaf, column fraction 0.9, L2 penalty 2.0.
  CPU binary objective, six threads, seed 42, deterministic column-wise fitting.
- Early stopping: internal tune binary logloss only, patience 50. The ultimate
  configuration-selection criterion is canonical tune Macro F0.5, not logloss.
- One threshold procedure: global probabilities >= threshold, coarse step 0.05
  over [0,1], followed by step 0.005 within one coarse step of the best point.
  An above-one threshold permits all-empty predictions. Equal Macro F0.5 prefers
  the larger threshold; equal model scores prefer the first, smaller model.
- The selected fast threshold score must equal the canonical frozen evaluator's
  score exactly. Zero, one and multiple predictions remain possible.
- The development gate requires >=0.01 absolute Macro F0.5 gain over the unchanged
  matcher on the same tune queries. Additional safeguards limit absolute micro
  precision loss to 0.02 and singleton-correct-rate loss to 0.05. Country results
  are diagnostics reviewed before validation, never learned inputs or decision
  rules. France has no training labels and remains unseen.
- The selected model is not refitted on tune queries. Its best-iteration text
  artifact, checksum, feature order, versions, configuration hashes, split
  membership hashes and threshold are frozen before any validation labels load.
- Validation refuses an existing report or an existing start marker. There is
  no force flag. A failed attempt retains its marker and cannot silently become
  another validation experiment.

## Reproduction

```bash
python -m pip install -r requirements.txt
python scripts/run_supervised.py --development
python scripts/run_supervised.py --validation  # only after the development gate passes
python -m pytest -q
```

The development command also refuses to overwrite a completed experiment report.
Run reports live under ignored `data/interim/matching/`; the compact selected
model and metadata live under `output/supervised_v1/`. Neither existing official
submission file is opened for writing. No feature matrix is saved to disk.

## Measured development result

The Phase 6 development gate **passed**. The selected 63-leaf model scored
**0.9259311292573912**, against **0.8599705764462935**
for Phase 5 on the identical 7,994 tune queries: **+0.0659605528110977** absolute Macro F0.5.
The global probability threshold is **0.660**. Both attempted configurations are listed below; no third configuration was tried.

Environment: Python 3.14.7, LightGBM **4.7.0**, NumPy 2.5.3, scikit-learn 1.9.1, SciPy 1.18.1.
LightGBM is pinned in `requirements.txt`. Normal dependency installation succeeded after a slow wheel download.

### Query split and class balance

| Partition | Queries | Screened pairs | Positives | Negatives | Positive rate |
| --- | --- | --- | --- | --- | --- |
| train | 32176 | 794459 | 104450 | 690009 | 13.14731157% |
| tune | 7994 | 198913 | 26105 | 172808 | 13.12382801% |

- Train membership SHA256: `1ba97c7b6644d736e8bf75305d34334409771978eaf1855cd9887d33cff62163`.
- Tune membership SHA256: `55831052fb4487447642c18ceb032fc3c53045ab3c7503a3a11706629c88aeb8`.
- Frozen source split fingerprint: `1dd67d77fee74aa0b64477e10f09377ac22854f5eb13780f3aacc4baf94e3645`.

Phase 5 had previously been tuned on the whole 40,170-query sample; this internal holdout is new for LightGBM. Frozen validation remains outside both Phase 6 fitting and selection.

### Candidate regression

| Stage | Pairs | Mean/query | Median | p95 | Max | Recall | Oracle Macro F0.5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Phase 3B | 3454436 | 85.995419467264 | 100 | 100 | 100 | 0.948549905296 | 0.980570186159 |
| Phase 5 screen | 993372 | 24.729200896191 | 21 | 51 | 100 | 0.940238957747 | 0.977372631817 |

All candidate statistics and positive counts matched the saved Phase 5 report. The reproduced deterministic whole-sample score was exactly 0.8563572538832931.
The only materialized feature matrix was 993,372 × 42 float32 values (159.16 MiB) in RAM. All positives and negatives were used; no matrix was persisted.

### Two fixed configurations

| Model | Best iteration | Tune logloss | Threshold | Tune Macro F0.5 | Fit + evaluation seconds |
| --- | --- | --- | --- | --- | --- |
| leaves31 | 500 | 0.0363512932 | 0.705 | 0.9235863645259758 | 64.53 |
| leaves63 | 500 | 0.0350768674 | 0.66 | 0.9259311292573912 | 77.26 |

Both reached the predeclared 500-round cap; the cap was not extended. The coarse-to-fine threshold score agreed exactly with the canonical evaluator for both models. Logloss was used only for early stopping; the larger tune Macro F0.5 selected `leaves63`.

### Same-query comparison

| Metric | Phase 5 | Selected LightGBM |
| --- | --- | --- |
| Macro F0.5 | 0.8599705764462935 | 0.9259311292573912 |
| TP | 20787 | 24126 |
| FP | 1147 | 695 |
| FN | 6961 | 3622 |
| Micro precision | 0.947706756633537 | 0.9719995165384151 |
| Micro recall | 0.7491350727980395 | 0.8694680697707943 |
| Correct singletons | 373 | 410 |
| Singleton false merges | 93 | 56 |
| Singleton queries | 466 | 466 |
| Zero predictions | 611 | 537 |
| One prediction | 1193 | 808 |
| Multiple predictions | 6190 | 6649 |
| Exact-match entities | 3330 | 5000 |

### Country diagnostics on tune

| Country | Matcher | Queries | Macro F0.5 | Precision | Recall | TP / FP / FN | Correct singletons / total | Singleton false merges |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| India | Phase 5 | 3108 | 0.8051252467 | 0.89204808 | 0.71396855 | 7718 / 934 / 3092 | 96 / 170 | 74 |
| India | LightGBM | 3108 | 0.8928990963 | 0.95923622 | 0.82719704 | 8942 / 380 / 1868 | 140 / 170 | 30 |
| US | Phase 5 | 4886 | 0.8948578636 | 0.98396326 | 0.77157870 | 13069 / 213 / 3869 | 277 / 296 | 19 |
| US | LightGBM | 4886 | 0.9469429095 | 0.97967611 | 0.89644586 | 15184 / 315 / 1754 | 270 / 296 | 26 |

Before validation, these diagnostics were reviewed: India improved in Macro F0.5, precision, recall, and singleton handling. Overall precision increased and singleton false merges fell from 93 to 56. US singleton false merges rose from 19 to 26 (correct rate 93.58% to 91.22%); this modest regression is recorded, not tuned away. No catastrophic precision or singleton regression was observed. Country never enters the feature matrix or threshold rule.

### Source diagnostics on tune

| Source | Matcher | TP | FP | FN | Precision | Recall |
| --- | --- | --- | --- | --- | --- | --- |
| source2 | Phase 5 | 10166 | 684 | 3194 | 0.9369585253 | 0.7609281437 |
| source2 | LightGBM | 11639 | 355 | 1721 | 0.9704018676 | 0.8711826347 |
| source3 | Phase 5 | 10621 | 463 | 3767 | 0.9582280765 | 0.7381845983 |
| source3 | LightGBM | 12487 | 340 | 1901 | 0.9734934123 | 0.8678760078 |

### Frozen artifact

- Model: `output/supervised_v1/model.txt` (3,519,918 bytes; compact text, no Git LFS).
- SHA256: `744aa05566e3d77f0c10c9d01a450a8bf518fd3142b97efe87629b41d52b9e75`.
- Metadata: `output/supervised_v1/metadata.json`; includes exact feature order, all model parameters and seeds, split salt and membership hashes, selected iteration and threshold, LightGBM version, configuration hashes and development gate result.
- Save/load reproduced every tune float32 probability exactly. Inference checks model checksum, feature order, shape and LightGBM version.
- No refit on tune queries. This artifact was frozen before starting the single validation confirmation.

### Selected model feature importance by gain

| Feature | Gain | Share |
| --- | --- | --- |
| blocking_aggregate_rarity | 1992391.947 | 33.8745% |
| blocking_support_families | 1489346.320 | 25.3218% |
| address_token_set_similarity | 609530.199 | 10.3632% |
| name_char_similarity | 263596.530 | 4.4817% |
| number_jaccard | 185203.860 | 3.1488% |
| target_numbers_in_query | 181201.789 | 3.0808% |
| blocking_min_key_frequency | 177309.554 | 3.0146% |
| address_token_containment | 166472.504 | 2.8304% |
| name_token_jaccard | 147496.060 | 2.5077% |
| name_token_set_similarity | 118309.957 | 2.0115% |
| query_numbers_in_target | 83367.795 | 1.4174% |
| name_length_ratio | 76027.203 | 1.2926% |
| name_token_intersection | 59964.317 | 1.0195% |
| address_char_similarity | 55914.621 | 0.9507% |
| channel_name_token_x_address_number | 52344.887 | 0.8900% |
| address_token_jaccard | 39392.454 | 0.6697% |
| name_script_mismatch | 37699.384 | 0.6410% |
| address_token_intersection | 31660.917 | 0.5383% |
| address_length_ratio | 21359.549 | 0.3632% |
| blocking_support_channels | 20030.702 | 0.3406% |
| target_number_count | 14023.165 | 0.2384% |
| name_token_containment | 13813.408 | 0.2349% |
| source_is_s3 | 8123.887 | 0.1381% |
| blocking_rank | 6929.686 | 0.1178% |
| name_folded_compact_equal | 3881.582 | 0.0660% |
| channel_name_x_address_word | 3864.972 | 0.0657% |
| query_number_count | 3695.162 | 0.0628% |
| channel_name_token_pair | 3539.986 | 0.0602% |
| blocking_exact_name | 2821.875 | 0.0480% |
| channel_name_exact | 2524.902 | 0.0429% |
| channel_name_folded_compact | 1735.090 | 0.0295% |
| channel_address_rare_word | 1667.453 | 0.0283% |
| channel_name_folded | 1561.719 | 0.0266% |
| shared_number_count | 1511.008 | 0.0257% |
| channel_address_number_x_word | 952.023 | 0.0162% |
| channel_address_exact | 889.937 | 0.0151% |
| number_set_equal | 880.010 | 0.0150% |
| channel_name_rare_token | 453.691 | 0.0077% |
| name_folded_equal | 71.064 | 0.0012% |
| name_canonical_equal | 67.550 | 0.0011% |
| address_canonical_equal | 54.526 | 0.0009% |
| target_address_empty | 0.000 | 0.0000% |

Importance is diagnostic only. All 42 features, including zero-gain features, remain in the schema.

### Development resources

Runner time: **469.29 s**; main-process peak RSS: **4277.10 MiB**; minimum observed system available memory: **4384.26 MiB**. These timings exclude dependency installation and tests.
Python forkserver workers were not accounted for by `RUSAGE_CHILDREN` (reported zero); worker/aggregate peak RSS is unavailable, and zero is not a worker-memory measurement. The system-memory snapshots include all processes but are not a continuous peak measurement.

| Checkpoint | Free disk GiB |
| --- | --- |
| start | 17.581543 |
| candidate_collection_complete | 17.577339 |
| before_training_leaves31 | 17.577309 |
| after_training_leaves31 | 17.577168 |
| before_training_leaves63 | 17.577141 |
| after_training_leaves63 | 17.577408 |
| before_model_save | 17.577301 |
| finish | 17.573719 |

Minimum observed free disk: **17.573719 GiB**. No unsafe disk or memory condition occurred.

## Single frozen validation confirmation

On all **220,410** frozen validation queries, the unchanged model and threshold scored **0.9253015854145195**, compared with Phase 5's **0.853761517044282** (reported as 0.85376): **+0.07154006837023741** absolute Macro F0.5.
Exactly one Phase 6 validation run was executed, after the development gate and manual country review. No validation-based model, feature, threshold, or rule change was made. The report and exclusive start marker prevent a second run. This result completes Phase 6.

| Metric | Phase 5 | Frozen LightGBM |
| --- | --- | --- |
| Macro F0.5 | 0.853761517044282 | 0.9253015854145195 |
| TP | 567270 | 659664 |
| FP | 35783 | 19524 |
| FN | 194322 | 101928 |
| Micro precision | 0.940663590099046 | 0.9712539090796657 |
| Micro recall | 0.7448476349541486 | 0.8661645605521067 |
| Correct singletons | 9483 | 10953 |
| Singleton false merges | 2877 | 1407 |
| Singleton queries | 12360 | 12360 |
| Zero predictions | 16499 | 14504 |
| One prediction | 34411 | 22609 |
| Multiple predictions | 169500 | 183297 |
| Exact-match entities | 90637 | 137609 |

### Validation candidate regression

| Stage | Pairs | Mean/query | Median | p95 | Max | Recall | Oracle Macro F0.5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Phase 3B | 18972356 | 86.077564538814 | 100 | 100 | 100 | 0.948720574796 | 0.980770729234 |
| Phase 5 screen | 5462251 | 24.782228574021 | 21 | 51 | 100 | 0.940141177954 | 0.977474472642 |

All candidate distributions, recalls, oracle scores, and source-positive counts exactly reproduce the saved Phase 5 validation report. Four queries have no candidates. The prefilter and `candidate_pairs.tsv` semantics remain unchanged; predictions are always a subset of screened candidates.

### Validation country diagnostics

| Country | Matcher | Macro F0.5 | Precision | Recall | TP / FP / FN | Correct singletons / total | Singleton false merges |
| --- | --- | --- | --- | --- | --- | --- | --- |
| India | Phase 5 | 0.8000131236 | 0.87986084 | 0.71198275 | 216986 / 29628 / 87777 | 2628 / 4851 | 2223 |
| India | LightGBM | 0.8969732142 | 0.95782488 | 0.82775796 | 252270 / 11108 / 52493 | 4143 / 4851 | 708 |
| US | Phase 5 | 0.8894720168 | 0.98273197 | 0.76677269 | 350284 / 6155 / 106545 | 6855 / 7509 | 654 |
| US | LightGBM | 0.9441229907 | 0.97975999 | 0.89178664 | 407394 / 8416 / 49435 | 6810 / 7509 | 699 |

### Validation source diagnostics

| Source | Matcher | TP | FP | FN | Precision | Recall |
| --- | --- | --- | --- | --- | --- | --- |
| source2 | Phase 5 | 277859 | 20348 | 90452 | 0.9317655186 | 0.7544140685 |
| source2 | LightGBM | 319516 | 9916 | 48795 | 0.9698997062 | 0.8675168540 |
| source3 | Phase 5 | 289411 | 15435 | 103870 | 0.9493678776 | 0.7358885886 |
| source3 | LightGBM | 340148 | 9608 | 53133 | 0.9725294205 | 0.8648981263 |

### Validation resources

Runner time: **875.89 s**; main-process peak RSS: **4558.42 MiB**; minimum observed system available memory: **5317.64 MiB**. Worker/aggregate RSS remains unavailable for the reason above. Features were scored and discarded batch by batch.

| Checkpoint | Free disk GiB |
| --- | --- |
| start | 17.572525 |
| streaming_inference_complete | 17.556995 |
| finish | 17.556980 |

Minimum observed free disk: **17.556980 GiB**, above the 10.5 GiB stop guard and 10 GiB required reserve. Development plus validation runner time was **1345.18 s**; setup and tests are additional.

## Limits and deferred work

- The selected model learns from only 32,176 development queries. Both configurations hit their fixed round cap; no additional capacity search or training was performed.
- US validation precision declined slightly (98.2732% to 97.9760%), and singleton false merges rose from 654 to 699. The larger recall gain still improved its Macro F0.5. India improved substantially but remains below US performance.
- Candidate recall is fixed at about 94.01% on validation; unavailable true matches remain irrecoverable. There is no candidate-generation or prefilter change in this phase.
- France remains unseen in challenge training. Synthetic tests cover open country labels, but this is not a claim about performance on France. No test inference or leaderboard submission occurred.
- Further threshold/decision strategies, larger training, candidate or feature changes, and test/submission work are deferred to later explicitly authorized phases. No Phase 7 work was started.

## Reproducibility artifacts and preservation

- Detailed local run reports: `data/interim/matching/supervised-v1_development_report.json` and `supervised-v1_validation_report.json`. They remain ignored derived diagnostics, not feature caches.
- The compact model and metadata are committed under `output/supervised_v1/`; raw data, stores, indexes, and generated matrices are not staged.
- Existing `output/candidate_pairs.tsv` SHA256 remains `bd0cd868dfaa017e047637994ef38fa107d4c28e50e63f575d7512fabcb77394`.
- Existing `output/matching_results.tsv` SHA256 remains `4cb9a2327cd6cecb6a131dba5b7756a2c4d8084cfa212dc7a6efe383fc1cdaee`.
- The single shared Phase 5 code change is the optional feature callback; existing invocations do not use it. No candidate, feature, rule, frozen split, or submission-format code was rewritten.

## Final checks

`python -m pytest -q`: **218 passed** in 20.14 s after the frozen validation run,
including all 210 existing tests and eight focused supervised tests. Coverage
includes query partition isolation, schema/order and NaN handling, exact screen
reuse, threshold abstention and multiple predictions, empty truths, canonical
metric parity, model reload, validation rerun prevention, and open country labels.
One harmless scikit-learn feature-name warning comes from the synthetic
classifier-versus-Booster probability comparison. LightGBM also emitted an
`eval_set` deprecation warning during training; both fits completed successfully.
`git diff --check` passed. No further optimization followed validation.
