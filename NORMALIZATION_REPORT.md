# Phase 2 — Conservative Normalization Report

This phase adds a conservative, Unicode-safe canonical normalizer for business
names and addresses, plus a development-only analysis of its effect. It does not
implement blocking, candidate generation, fuzzy matching, similarity features,
models, thresholds or predictions. Raw challenge files, official resources, the
frozen split, its fingerprint and the evaluation protocol are unchanged.

**Validation labels were not used.** No normalization rule was designed, chosen or
tuned using validation labels. Every statistic here comes from the frozen
development partition (split fingerprint
`1dd67d77fee74aa0b64477e10f09377ac22854f5eb13780f3aacc4baf94e3645`): development
Source 1 records, development-owned Source 2/3 records, and labels of development
Source 1 entities only. The streaming truth reader parses every row, but it drops a
validation row before storing or counting it. Rule design also used a
development-only character census and a small sample of development links.

## Reproduction

```bash
python scripts/analyze_normalization.py          # writes data/interim/normalization_analysis.json
python -m pytest -q
```

The analysis first checks the frozen manifests, the committed fingerprint and the
SHA-256 hashes of all four raw training inputs against the split metadata. It then
streams each source once, in 20,000-record batches, across six worker processes.
Each record's representations are kept only as signed 64-bit BLAKE2b keys, in a
temporary SQLite database that is deleted afterwards; no normalized copy of the
data is written. Keys are compared for equality and grouped for collision
statistics. With roughly 4.7 million distinct values per source, the chance of any
64-bit key collision is about 10⁻⁶, so it cannot move any reported count. A second
streaming pass recovers the few strings shown as examples.

**Runtime:** 1,496.6 s (about 25 minutes) on 6 worker processes:

| Stage | Time |
| --- | --- |
| Raw-input hash verification | ≈20 s |
| Source profiling (S1 / S2 / S3) | ≈1.7 / 4.5 / 5.3 min |
| Development link loading and verification | ≈1.5 min |
| Link equality | ≈40 s |
| Collision grouping | ≈7.5 min |
| Example recovery pass | ≈3.5 min |

The main process was observed at about 150 MB RSS and each worker at about 25 MB.
Peak memory was not formally recorded.

## Canonical normalization rules

Implementation: `src/preprocessing/normalization.py` (version `canonical-v1`).
`normalize_name` and `normalize_address` both apply the same `canonicalize`
function, in this order:

1. **Corruption markers.** U+FFFD, and its Latin-1 mojibake `ï¿½` / `Ï¿½`, become a
   separator.
2. **Apostrophes.** Apostrophe-like elision marks are deleted, not split:
   `' ` ´ ‘ ’ ‛ ʼ ＇ ｀`. So `CANDY'S` → `candys` and `GOVERNOR`S` → `governors`.
   Deletion runs both before and after step 3, because NFKC can create `ʼ` from `ŉ`.
3. **Unicode normalization.** NFKC, then Unicode `casefold()`, then NFKC again.
   Full-width and compatibility forms fold (`ＡＢＣ１２３` → `abc123`, `ﬁ` → `fi`,
   `①` → `1`), and `ß` → `ss`.
4. **Character classes.**
   - Letters (L\*), combining marks (M\*) and digits/numerals (N\*) of every script
     are kept.
   - `&` is kept as its own token.
   - Invisible format characters (Cf: ZWNJ, ZWJ, zero-width space, soft hyphen) are
     deleted without splitting the word.
   - Every other character becomes a separator: punctuation, symbols, controls and
     all whitespace.
5. **Final form.** A last NFKC composition, then separators collapse to single
   spaces with no leading or trailing space.

Properties:

| Property | Behavior |
| --- | --- |
| Determinism, idempotence | Deterministic; idempotent on every development field (see Unicode diagnostics). |
| Empty input | An empty string stays empty. |
| Literal null-like strings | `NA`, `NULL`, `None`, `nan`, `<NULL>` stay text (`na`, `null`, …); they are never treated as missing. |
| Non-string input | Raises `TypeError` rather than silently becoming `"none"`. |
| Numbers | Digits are never removed or rewritten. Punctuation between digits becomes a space (`15/43` → `15 43`), so number boundaries survive. Leading zeros are kept (`015115` stays distinct from `15115`). |
| Scripts | Devanagari, Tamil, Kannada, Telugu, Malayalam and accented Latin survive with their marks: `राम` → `राम`, `École` → `école`, `Straße` → `strasse`. |

Auxiliary views are kept separate and never replace the canonical form:

| View | Definition | Status |
| --- | --- | --- |
| `compact` | Canonical form with its spaces removed (`tri state` → `tristate`, `l l c` → `llc`). | Auxiliary. |
| `folded` | Canonical form with nonspacing marks removed **only where they attach to Latin letters** (`prívate ínc` → `private inc`). Devanagari and all other scripts, and undecomposable Latin letters (`ø`, `ł`, `æ`), are untouched. | Auxiliary and lossy. Retained because of the development measurements below. |
| `folded_compact` | `compact` applied to `folded`. | Auxiliary, diagnostic. |

Raw fields are never overwritten. `src/preprocessing/records.py` adds
`business_name_norm` and `business_address_norm` beside the untouched raw fields
(`normalize_record`, `iter_normalized_source`).

### Deliberately omitted transformations

Phase 2 replaced the earlier normalizer. That normalizer:
- deleted every non-ASCII letter through an ASCII-only regex, erasing Devanagari
  and other Indic names entirely;
- stripped all accents;
- turned the literal string `nan` into an empty string;
- rewrote `street` → `st` and similar words;
- optionally dropped legal suffixes.

None of these remain. The canonical form also deliberately does **not**:

| Omitted transformation | Reason |
| --- | --- |
| Expand or contract abbreviations (`rd`/`road`, `st`/`street`, `pvt`/`private`, US state codes) | These are semantic substitutions. `st` is ambiguous (street, saint, state), and the correct rules are country-specific: France is unseen in training. They belong in later auxiliary features, measured separately. |
| Remove legal suffixes or business words | They carry evidence (`Inc` vs `LLC`) and removing them raises collisions among generic names (see below). |
| Replace `&` with `and` | Not reversible, and language-specific (`et`, `und`, `और`). |
| Transliterate or translate | Devanagari ↔ Latin is a model-level decision with its own error profile. Deferred. |
| Strip leading zeros from house numbers | Observed on development links (`015115` vs `15115`), but it is a numeric-token feature, not text canonicalization. |
| Map non-ASCII digits to ASCII | No non-ASCII decimal digit (Nd) occurs anywhere in the development records, so a rule would be untested. |
| Repair mojibake beyond the U+FFFD marker (for example `Â\x80\x99`) | The corruption appears identically across sources, so repair would not recover equality and could introduce errors. |

## Development records and positive links analysed

| Source | Development records | Nonempty names | Nonempty addresses |
| --- | ---: | ---: | ---: |
| S1 | 1,986,411 | 1,986,411 | 1,986,411 |
| S2 | 4,531,619 | 4,531,619 | 4,379,385 |
| S3 | 4,757,870 | 4,757,870 | 4,599,536 |

Record counts equal the frozen development manifests; the script fails otherwise.
Development positive links are exactly **S2 = 3,325,308** and **S3 = 3,551,465**.
These equal the frozen split metadata, and every link joins a development S1
record and a development-owned target record. The script fails loudly on any
mismatch. No development link crosses countries (0 country mismatches). Among the
links, 148,826 S2 targets (4.48%) and 154,665 S3 targets (4.35%) have empty
addresses, so their address equality is impossible under any representation.

## Positive-link exact equality

Each cell counts genuine development links whose S1 and target representations are
identical. Empty values never count as equal.

| Field | Representation | S1→S2 count | S1→S2 % | S1→S3 count | S1→S3 % |
| --- | --- | ---: | ---: | ---: | ---: |
| Name | raw | 157,087 | 4.7240% | 161,609 | 4.5505% |
| Name | canonical | 706,962 | 21.2600% | 782,907 | 22.0446% |
| Name | compact | 747,090 | 22.4668% | 817,731 | 23.0252% |
| Name | folded | 834,030 | 25.0813% | 925,986 | 26.0734% |
| Name | folded_compact | 875,311 | 26.3227% | 961,892 | 27.0844% |
| Address | raw | 3 | 0.0001% | 153,138 | 4.3120% |
| Address | canonical | 415,163 | 12.4849% | 153,479 | 4.3216% |
| Address | compact | 418,464 | 12.5842% | 153,486 | 4.3218% |
| Address | folded | 415,172 | 12.4852% | 153,479 | 4.3216% |
| Address | folded_compact | 418,473 | 12.5845% | 153,486 | 4.3218% |
| Name + address | raw | 1 | 0.0000% | 13 | 0.0004% |
| Name + address | canonical | 89,096 | 2.6793% | 118 | 0.0033% |
| Name + address | compact | 94,929 | 2.8547% | 120 | 0.0034% |
| Name + address | folded | 104,970 | 3.1567% | 147 | 0.0041% |
| Name + address | folded_compact | 111,067 | 3.3401% | 149 | 0.0042% |

The development raw rates (4.72% / 4.55% names, 0.0001% / 4.31% addresses)
reproduce the full-training Phase 1 figures.

By reference country:

| Link source | Country | Links | Raw name | Canonical name | Folded name | Raw address | Canonical address | Canonical name + address |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S1→S2 | India | 1,332,937 | 2.6681% | 14.7210% | 17.2614% | 0.0000% | 11.2539% | 1.5942% |
| S1→S2 | US | 1,992,371 | 6.0994% | 25.6348% | 30.3130% | 0.0002% | 13.3085% | 3.4053% |
| S1→S3 | India | 1,422,143 | 2.7489% | 16.6270% | 19.7657% | 4.1412% | 4.1652% | 0.0083% |
| S1→S3 | US | 2,129,322 | 5.7538% | 25.6630% | 30.2861% | 4.4260% | 4.4260% | 0.0000% |

## S2 versus S3 differences

- **Names behave alike.** Canonical name equality lifts both sources by 4.5–4.8×
  (S2 4.72% → 21.26%, S3 4.55% → 22.04%). Latin folding adds a further 3.8–4.0
  points to each. India links have much lower name equality than US links in both
  sources. Many India target names are in Devanagari (S2 242,320, S3 141,906) or
  another Indic script (S2 184,771, S3 108,605), and these can never equal an
  all-ASCII S1 name without transliteration.
- **Addresses differ sharply.** S2 addresses are mainly case/punctuation variants of
  S1 (often upper-case, often truncated). Canonicalization lifts S2 address
  equality from 3 links to 415,163 (12.48%). S3 addresses are already in S1's case,
  but they reorder components, expand state codes (`IL` → `Illinois`) and drop or
  corrupt parts. Canonicalization barely moves them (4.312% → 4.322%); this ceiling
  is structural, not a normalization failure.
- **Joint name+address equality.** Reachable for S2 (2.68% canonical, 3.34%
  folded_compact) but essentially absent for S3 (118–149 links), because S3
  addresses rarely equal S1's even when the names do.
- **Whitespace.** S2 has far more irregular address whitespace (96,667 vs 4,414).
  Irregular name whitespace is similar (S2 499,103, S3 517,963).
- **Accents.** S3 has slightly more accented-Latin names (295,476 vs 261,057).

## Collision statistics

A "bucket" is the set of records within one source sharing a representation key.
"Records in duplicated buckets" counts records whose key is shared with at least
one other record in the same source. Empty values are excluded.

### Names

| Representation | Source | Nonempty records | Distinct values | Records in duplicated buckets | Duplicated share | Largest bucket | Same-key record pairs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | S1 | 1,986,411 | 1,401,978 | 742,480 | 37.38% | 227 | 6,633,703 |
| raw | S2 | 4,531,619 | 3,979,162 | 767,075 | 16.93% | 283 | 3,321,141 |
| raw | S3 | 4,757,870 | 4,205,599 | 782,103 | 16.44% | 370 | 3,975,284 |
| canonical | S1 | 1,986,411 | 1,386,879 | 759,873 | 38.25% | 227 | 7,182,318 |
| canonical | S2 | 4,531,619 | 3,654,504 | 1,229,447 | 27.13% | 472 | 7,006,455 |
| canonical | S3 | 4,757,870 | 3,885,516 | 1,227,875 | 25.81% | 463 | 7,415,501 |
| compact | S1 | 1,986,411 | 1,382,657 | 765,179 | 38.52% | 227 | 7,294,654 |
| compact | S2 | 4,531,619 | 3,628,166 | 1,267,109 | 27.96% | 551 | 7,549,759 |
| compact | S3 | 4,757,870 | 3,858,485 | 1,265,555 | 26.60% | 548 | 7,975,191 |
| folded | S1 | 1,986,411 | 1,386,879 | 759,873 | 38.25% | 227 | 7,182,318 |
| folded | S2 | 4,531,619 | 3,585,302 | 1,329,656 | 29.34% | 485 | 7,853,538 |
| folded | S3 | 4,757,870 | 3,804,653 | 1,342,263 | 28.21% | 474 | 8,381,168 |
| folded_compact | S1 | 1,986,411 | 1,382,657 | 765,179 | 38.52% | 227 | 7,294,654 |
| folded_compact | S2 | 4,531,619 | 3,556,404 | 1,370,755 | 30.25% | 563 | 8,442,551 |
| folded_compact | S3 | 4,757,870 | 3,775,253 | 1,383,015 | 29.07% | 562 | 8,990,894 |

### Addresses (empty addresses excluded)

| Representation | Source | Nonempty records | Distinct values | Records in duplicated buckets | Duplicated share | Largest bucket | Same-key record pairs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | S1 | 1,986,411 | 1,921,486 | 100,316 | 5.05% | 13 | 123,955 |
| raw | S2 | 4,379,385 | 3,909,541 | 847,003 | 19.34% | 18 | 603,722 |
| raw | S3 | 4,599,536 | 4,175,579 | 766,737 | 16.67% | 27 | 552,836 |
| canonical | S1 | 1,986,411 | 1,921,129 | 101,020 | 5.09% | 13 | 124,327 |
| canonical | S2 | 4,379,385 | 3,864,054 | 924,701 | 21.11% | 18 | 671,724 |
| canonical | S3 | 4,599,536 | 4,160,927 | 790,189 | 17.18% | 35 | 590,581 |
| compact | S1 | 1,986,411 | 1,919,773 | 103,638 | 5.22% | 13 | 125,828 |
| compact | S2 | 4,379,385 | 3,862,376 | 927,406 | 21.18% | 18 | 674,533 |
| compact | S3 | 4,599,536 | 4,158,180 | 794,204 | 17.27% | 35 | 598,337 |
| folded | S1 | 1,986,411 | 1,921,129 | 101,020 | 5.09% | 13 | 124,327 |
| folded | S2 | 4,379,385 | 3,864,043 | 924,717 | 21.12% | 18 | 671,743 |
| folded | S3 | 4,599,536 | 4,160,921 | 790,193 | 17.18% | 35 | 590,604 |
| folded_compact | S1 | 1,986,411 | 1,919,773 | 103,638 | 5.22% | 13 | 125,828 |
| folded_compact | S2 | 4,379,385 | 3,862,365 | 927,422 | 21.18% | 18 | 674,552 |
| folded_compact | S3 | 4,599,536 | 4,158,174 | 794,208 | 17.27% | 35 | 598,360 |

### Name + address

S1 (name, address) is unique under every representation: 1,986,411 distinct
values and a largest bucket of 1. In S2 and S3, joint duplicates grow from 1.04% /
0.72% (raw) to 2.60% / 1.84% (canonical) and 3.27% / 2.39% (folded_compact), and
buckets never exceed 5 records.

| Representation | Source | Nonempty records | Distinct values | Records in duplicated buckets | Duplicated share | Largest bucket | Same-key record pairs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | S1 | 1,986,411 | 1,986,411 | 0 | 0.00% | 1 | 0 |
| raw | S2 | 4,379,385 | 4,356,282 | 45,477 | 1.04% | 5 | 23,855 |
| raw | S3 | 4,599,536 | 4,582,779 | 33,086 | 0.72% | 4 | 17,201 |
| canonical | S1 | 1,986,411 | 1,986,411 | 0 | 0.00% | 1 | 0 |
| canonical | S2 | 4,379,385 | 4,321,278 | 113,999 | 2.60% | 5 | 60,395 |
| canonical | S3 | 4,599,536 | 4,556,452 | 84,606 | 1.84% | 4 | 44,696 |
| compact | S1 | 1,986,411 | 1,986,411 | 0 | 0.00% | 1 | 0 |
| compact | S2 | 4,379,385 | 4,317,653 | 121,022 | 2.76% | 5 | 64,253 |
| compact | S3 | 4,599,536 | 4,553,562 | 90,227 | 1.96% | 4 | 47,747 |
| folded | S1 | 1,986,411 | 1,986,411 | 0 | 0.00% | 1 | 0 |
| folded | S2 | 4,379,385 | 4,310,533 | 134,794 | 3.08% | 5 | 71,860 |
| folded | S3 | 4,599,536 | 4,546,777 | 103,313 | 2.25% | 4 | 55,051 |
| folded_compact | S1 | 1,986,411 | 1,986,411 | 0 | 0.00% | 1 | 0 |
| folded_compact | S2 | 4,379,385 | 4,306,244 | 143,085 | 3.27% | 5 | 76,447 |
| folded_compact | S3 | 4,599,536 | 4,543,382 | 109,888 | 2.39% | 4 | 58,667 |

### By country

Largest buckets and duplicated records per country, for the raw, canonical and
most lenient views:

| Field | Representation | Source | Country | Distinct values | Records in duplicated buckets | Largest bucket |
| --- | --- | --- | --- | ---: | ---: | ---: |
| name | raw | S1 | India | 513,550 | 342,458 | 93 |
| name | raw | S1 | US | 889,008 | 399,430 | 227 |
| name | raw | S2 | India | 1,524,678 | 391,606 | 72 |
| name | raw | S2 | US | 2,459,040 | 371,815 | 283 |
| name | raw | S3 | India | 1,658,421 | 350,453 | 157 |
| name | raw | S3 | US | 2,553,042 | 426,681 | 370 |
| name | canonical | S1 | India | 511,367 | 345,473 | 93 |
| name | canonical | S1 | US | 876,102 | 413,792 | 227 |
| name | canonical | S2 | India | 1,441,313 | 522,607 | 115 |
| name | canonical | S2 | US | 2,217,625 | 703,500 | 472 |
| name | canonical | S3 | India | 1,556,737 | 494,049 | 166 |
| name | canonical | S3 | US | 2,335,309 | 728,321 | 463 |
| name | folded_compact | S1 | India | 511,245 | 345,652 | 93 |
| name | folded_compact | S1 | US | 872,008 | 418,913 | 227 |
| name | folded_compact | S2 | India | 1,418,779 | 557,323 | 162 |
| name | folded_compact | S2 | US | 2,142,207 | 810,013 | 563 |
| name | folded_compact | S3 | India | 1,523,959 | 539,838 | 179 |
| name | folded_compact | S3 | US | 2,258,035 | 837,542 | 562 |
| address | raw | S1 | India | 772,283 | 36,137 | 11 |
| address | raw | S1 | US | 1,149,203 | 64,179 | 13 |
| address | raw | S2 | India | 1,584,110 | 324,161 | 15 |
| address | raw | S2 | US | 2,325,431 | 522,842 | 18 |
| address | raw | S3 | India | 1,709,512 | 246,887 | 27 |
| address | raw | S3 | US | 2,466,067 | 519,850 | 15 |
| address | canonical | S1 | India | 771,926 | 36,841 | 11 |
| address | canonical | S1 | US | 1,149,203 | 64,179 | 13 |
| address | canonical | S2 | India | 1,573,656 | 342,458 | 15 |
| address | canonical | S2 | US | 2,290,398 | 582,243 | 18 |
| address | canonical | S3 | India | 1,697,543 | 265,641 | 35 |
| address | canonical | S3 | US | 2,463,384 | 524,548 | 15 |
| address | folded_compact | S1 | India | 770,571 | 39,457 | 11 |
| address | folded_compact | S1 | US | 1,149,202 | 64,181 | 13 |
| address | folded_compact | S2 | India | 1,572,102 | 344,950 | 15 |
| address | folded_compact | S2 | US | 2,290,263 | 582,472 | 18 |
| address | folded_compact | S3 | India | 1,694,882 | 269,523 | 35 |
| address | folded_compact | S3 | US | 2,463,292 | 524,685 | 15 |

### Largest buckets and most-merged keys (illustrative)

These are the largest bucket and most-merged keys per source. "Most merged" means
the most distinct raw variants, or for `folded` the most distinct canonical
variants. At most four variants are shown.

| Source / view | Key | Records | Kind | Variants merged |
| --- | --- | ---: | --- | --- |
| S1 name canonical | `primary care group` | 227 | largest | `Primary Care Group` (the single raw form) |
| S1 name canonical | `main street grill inc` | 9 | most merged | `Main Street Grill Inc.`, `Main Street Grill! Inc`, `Main Street Grill!, Inc`, `Main Street Grill Inc` |
| S1 address canonical | `108 norle street college twp pa` | 13 | largest | `108 Norle Street, College Twp, PA` |
| S1 address canonical | `gr flr 31 greenacre union park road no 5 khar west mumbai maharashtra` | 2 | most merged | two punctuation variants of the same Mumbai address |
| S2 name canonical | `physical therapy` | 472 | largest | `Physical Therapy`, `Physical  Therapy`, `PHYSICAL THERAPY`, `Physical (Therapy)` |
| S2 name canonical | `meridian inc` | 116 | most merged | `Meridian Inc`, `Meridian, Inc.`, `Meridian  Inc.`, `MERIDIAN  INC` |
| S2 name folded | `primary care` | 485 | largest | `primary care`, `primary cáre` |
| S2 name folded | `urgent care clinic inc` | 44 | most merged | `urgent care clinic inc`, `urgent care clínic inc`, `urgent care clinic ínc`, `urgent cáre clínic inc` |
| S2 address canonical | `842 38 caledonai township wi` | 18 | largest | `842 38, CALEDONAI TOWNSHIP, WI` |
| S2 address canonical | `no 5 chennai tamil nadu` | 12 | most merged | `NO.5, …`, `NO. 5, …`, `NO 5, …`, `NO.##5, CHENNAI, Tamil Nadu` |
| S3 name canonical | `primary care` | 463 | largest | `Primary Care`, `Primary  Care`, `Primary-Care`, `Primary (Care)` |
| S3 name canonical | `meridian inc` | 124 | most merged | `*** Meridian Inc`, `Meridian-Inc.`, `Meridian-Inc`, `Meridian Inc` |
| S3 name folded | `pediatric dental` | 474 | largest | `pediatric dental`, `pediatric déntal`, `pédiatric dental` |
| S3 address canonical | `no 1 chennai tn` | 35 | largest | `No:1, Chennai, TN`, `No.1, …`, `No 1, …`, `No. 1, Chennai, TN` |
| S3 address canonical | `no 3 bangalore ka` | 17 | most merged | `No. ##3, …`, `No.3, …`, `No. #3, …`, `No: 3, Bangalore, KA` |

Every inspected merge is a case, spacing, punctuation, decoration (`***`, `()`) or
accent-noise variant of the same string. No merge joined textually unrelated
strings. The large buckets come from generic business names and from short,
truncated addresses, and those strings already collide in raw form. Folding
merges nothing in S1: its names are pure ASCII, and only 490 S1 addresses change,
all through mojibake letters such as `Â`.

## Collision tradeoff: equal-key cross pairs

Within-source collision growth overstates the danger, because S2/S3 intentionally
hold several records per business. The relevant measure is the cross-source one:
how many same-country (S1, target) record pairs share a key, and what fraction of
those pairs are genuine development links. This is a diagnostic of key purity; no
candidate pairs were generated or stored.

| Key | Representation | S1×S2 equal-key pairs | True links | True-link fraction | S1×S3 equal-key pairs | True links | True-link fraction |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| name | raw | 3,189,340 | 157,087 | 0.0493 | 3,608,611 | 161,609 | 0.0448 |
| name | canonical | 8,380,460 | 706,962 | 0.0844 | 9,239,096 | 782,907 | 0.0847 |
| name | compact | 8,791,350 | 747,090 | 0.0850 | 9,626,715 | 817,731 | 0.0849 |
| name | folded | 9,413,440 | 834,030 | 0.0886 | 10,432,726 | 925,986 | 0.0888 |
| name | folded_compact | 9,842,395 | 875,311 | 0.0889 | 10,838,615 | 961,892 | 0.0887 |
| address | raw | 3 | 3 | 1.0000 | 181,461 | 153,138 | 0.8439 |
| address | canonical | 498,923 | 415,163 | 0.8321 | 181,923 | 153,479 | 0.8436 |
| address | compact | 504,059 | 418,464 | 0.8302 | 182,270 | 153,486 | 0.8421 |
| address | folded | 498,932 | 415,172 | 0.8321 | 181,923 | 153,479 | 0.8436 |
| address | folded_compact | 504,068 | 418,473 | 0.8302 | 182,270 | 153,486 | 0.8421 |
| name_address | raw | 1 | 1 | 1.0000 | 13 | 13 | 1.0000 |
| name_address | canonical | 89,096 | 89,096 | 1.0000 | 118 | 118 | 1.0000 |
| name_address | compact | 94,929 | 94,929 | 1.0000 | 120 | 120 | 1.0000 |
| name_address | folded | 104,970 | 104,970 | 1.0000 | 147 | 147 | 1.0000 |
| name_address | folded_compact | 111,067 | 111,067 | 1.0000 | 149 | 149 | 1.0000 |

Marginal true-link fraction of the equal-key pairs **added** by each step:

| Step | Name S2 | Name S3 | Address S2 | Address S3 |
| --- | --- | --- | --- | --- |
| raw → canonical | +549,875 / +5,191,120 = 0.106 | +621,298 / +5,630,485 = 0.110 | +415,160 / +498,920 = 0.832 | +341 / +462 = 0.738 |
| canonical → compact | 0.098 | 0.090 | 0.643 | **0.020** (+7 / +347) |
| canonical → folded | **0.123** | **0.120** | 9 / 9 | 0 / 0 |

Interpretation:

- **Canonical normalization.** The extra collisions are no less pure than raw
  equality: names added at 0.11 against a raw baseline of 0.049, addresses at
  0.83. It more than quadruples positive name equality and creates S2 address
  equality, while name-key purity improves from 0.049 to 0.084.
- **Name keys alone are weak.** The normalized name views pair each development S1
  record with 4.2 (canonical) to 5.5 (folded_compact) same-name targets per source,
  and only about 9% of those pairs are genuine. Generic names drive this: 37% of development S1 records share
  their raw name with another S1 record, and up to 227 share one name. A name-only
  exact key is unusable as a match decision, and uncapped it produces
  heavy-bucket blocks.
- **Latin folding is worth keeping as an auxiliary view.** Its added pairs are
  purer (0.12) than the canonical average. It recovers 127,068 S2 and 143,079 S3
  links broken by synthetic accent noise, and it provably does not touch Indic
  text. It stays out of the canonical form because French test data will carry
  genuine accents.
- **Compact views help names slightly** (≈0.09–0.10 marginal). For S3 addresses
  they are harmful: 347 extra pairs for 7 links. Do not use compact addresses for
  S3.
- **Address and name+address keys are high-purity.** Canonical address equality is
  83–84% pure. Canonical name+address equality was 100% pure on every development
  pair, but it covers at most 3.34% of S2 and 0.004% of S3 links.

## Unicode diagnostics

Counts of development fields (records) with each property. The final block holds
the correctness checks, all of which must be zero.

| Diagnostic | S1 name | S1 address | S2 name | S2 address | S3 name | S3 address |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `nonempty` | 1,986,411 | 1,986,411 | 4,531,619 | 4,379,385 | 4,757,870 | 4,599,536 |
| `non_ascii` | 0 | 502 | 688,148 | 431,210 | 545,987 | 428,830 |
| `devanagari` | 0 | 0 | 242,320 | 250,119 | 141,906 | 247,941 |
| `other_indic_script` | 0 | 0 | 184,771 | 180,257 | 108,605 | 180,144 |
| `accented_or_non_ascii_latin` | 0 | 502 | 261,057 | 1,079 | 295,476 | 936 |
| `format_character` (ZWNJ etc.) | 0 | 0 | 23,060 | 0 | 13,057 | 0 |
| `nfkc_changes_raw` | 0 | 12 | 0 | 21 | 0 | 14 |
| `corruption_marker` (`ï¿½`, U+FFFD) | 0 | 12 | 0 | 21 | 0 | 14 |
| `control_character` | 1 | 552 | 3 | 1,112 | 1 | 989 |
| `casefold_differs_from_lower` | 0 | 0 | 0 | 0 | 0 | 0 |
| `apostrophe` | 41,877 | 4,911 | 86,518 | 10,015 | 90,795 | 8,912 |
| `hyphen_or_dash` | 12,132 | 320,027 | 239,086 | 960,157 | 265,805 | 946,635 |
| `ampersand` | 100,798 | 17,525 | 188,408 | 36,655 | 195,653 | 33,093 |
| `irregular_whitespace` | 0 | 885 | 499,103 | 96,667 | 517,963 | 4,414 |
| `canonical_changes` | 1,986,411 | 1,986,411 | 4,089,570 | 4,379,385 | 4,455,515 | 4,599,536 |
| `folding_changes` | 0 | 490 | 261,057 | 1,058 | 295,476 | 922 |
| **`empty_after_normalization`** (nonempty → empty) | 0 | 0 | 0 | 0 | 0 | 0 |
| **`non_ascii_empty_after_normalization`** | 0 | 0 | 0 | 0 | 0 | 0 |
| **`canonical_not_idempotent`** | 0 | 0 | 0 | 0 | 0 | 0 |
| **`folding_not_idempotent`** | 0 | 0 | 0 | 0 | 0 | 0 |
| **`letters_or_digits_lost`** | 0 | 0 | 0 | 0 | 0 | 0 |
| **`devanagari_altered`** | 0 | 0 | 0 | 0 | 0 | 0 |
| **`folding_altered_indic`** | 0 | 0 | 0 | 0 | 0 | 0 |

Findings:

- Across 22,241,232 nonempty development fields, **none** became empty. Among
  2,094,677 non-ASCII fields, none became empty either, and none failed idempotence.
- **Letter and digit loss.** No letter or digit was lost in any field. For ASCII
  fields the check is independent of the implementation (lowercased
  `str.isalnum()` characters against the canonical tokens). For non-ASCII fields it
  compares the canonical L/N characters against an NFKC-casefold reference in
  canonical decomposition, excluding only the documented apostrophe and
  corruption-marker deletions.
- **Indic text survives intact.** All 882,286 Devanagari fields keep every
  Devanagari code point, including combining vowel signs, virama and nukta.
  Folding never changed an Indic code point (U+0900–U+0DFF).
- **NFKC matters only for the corruption markers in this data.** No development
  field contains full-width or other compatibility characters; NFKC changed only
  the 47 `ï¿½` address fields. Without the marker rule, NFKC would turn `½` into
  the fake digits `1 2`. Full-width handling is covered by unit tests only.
- **Case folding.** `casefold()` never differs from `lower()` in development data
  (no `ß`, `İ` or similar), so it is a safety guarantee here rather than a source
  of lift.
- **Zero-width non-joiners.** ZWNJ appears inside 36,117 Telugu/Kannada/Devanagari
  names. Deleting it joins the word rather than splitting it.

### Suspicious or noteworthy transformations

- **Mojibake.** `ï¿½` (47 address fields) is now a separator. C1 control characters
  from `Â\x80\x99`/`Â\x80\x93` sequences (2,653 address fields) become separators
  and leave an `â` letter fragment. They are consistent across sources, so equality
  is unaffected, but the `â` fragment is noise for token features.
- **Literal `<NULL>` / `NULL` address components** stay as the token `null`. They
  were observed in S2 samples but not counted. Phase 3 should treat `null` as a
  stop-token for address blocking rather than as missing data.
- **Genuine vulgar fractions** would become separate digits (`½` → `1 2`) under
  NFKC. None exist in development data outside the corruption markers.
- **Short partial addresses** (`No:1, Chennai, TN`) form the largest address
  buckets (up to 35). Canonicalization merges their punctuation variants
  (`No.1`, `No#1`, `No: 1`). These addresses are intrinsically ambiguous.
- **Decoration tokens.** Web-like fragments (`| www.x.com` → `www x com`) and
  handle prefixes (`@name` → `name`) survive as ordinary tokens. They are real text,
  but they will usually prevent exact name equality. The census found `|` in about
  14–15K and `@` in about 15–16K development names per target source.

## Recommendations for Phase 3 candidate generation (not implemented)

1. **Restrict blocks to the same country.** No development link crosses countries
   (0 of 6,876,773). Keep `country` an open label so France forms its own blocks.
2. **Never rely on exact keys for recall.** The most lenient exact name view reaches
   only 26.3% (S2) / 27.1% (S3) of links; addresses reach 12.6% / 4.3%. Phase 3
   needs token- or n-gram-level blocking over the canonical and `folded` views, and
   must measure blocking recall per source and per country against development
   links, following AGENTS.md.
3. **Cap or down-weight generic tokens and names.** Canonical name buckets reach
   472–563 records per source and give 4–5 same-name targets per S1 record at
   about 9% purity. Blocking keys need frequency caps or IDF weighting, and
   per-S1 candidate-count statistics (median, p95, max) must be tracked.
4. **Use `folded` rather than raw or accented forms for Latin keys.** It is the
   purest-per-added-pair view measured (0.12 marginal). Keep the canonical form for
   features, so genuine French accents remain available.
5. **Treat S2 and S3 addresses differently.** S2 addresses benefit from canonical
   equality (12.5%, 83% pure). S3 addresses need order-insensitive token sets and
   state/region-code handling, which reordering and `IL` → `Illinois` expansion
   demand. Avoid compact-address keys for S3 (0.020 marginal purity).
6. **Add numeric-token keys** (house/plot numbers, PIN/ZIP-like tokens) with
   leading-zero handling. Development links show `015115` vs `15115` and dropped
   leading digits (`8061` vs `061`). Use them as an auxiliary representation, not
   by editing the canonical form.
7. **Plan an address-driven path for non-Latin names.** 427,091 S2 and 250,511 S3
   names are in Indic scripts that can never equal an ASCII S1 name. Candidates for
   them must come from addresses or a later, measured transliteration view.
8. **Canonical name+address equality is a near-certain match signal** (100% of
   89,096 S2 and 118 S3 equal pairs were genuine). It is suitable for a
   high-precision candidate tier or feature, but covers under 3.4% of links.
9. **Treat `null`, `www`, `com` and single-letter tokens from dotted abbreviations**
   (`l l c`) as low-information tokens in blocking, or rely on the `compact` view
   for dotted abbreviations.

## Tests

The complete suite passes: **152 tests**. Normalization coverage includes:
- empty strings and whitespace collapse;
- punctuation, symbol and decoration handling;
- idempotence and determinism;
- case folding and full-width/NFKC forms;
- Devanagari, Tamil, Kannada and accented Latin preservation;
- composed versus decomposed accents and format-character deletion;
- the corruption-marker regression;
- numeric preservation;
- literal `NA`/`NULL`/`None`/`<NULL>` staying text;
- the absence of semantic substitutions;
- separation of the compact and folded views;
- raw-record immutability, including raw file bytes;
- a synthetic end-to-end analysis checking exact equality counts, development-only
  link selection, identical serial and parallel results, and a loud failure when
  link counts mismatch.
