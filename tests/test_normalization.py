from __future__ import annotations

import copy
import csv
import unicodedata

import pytest

from src.data.load import SOURCE_COLUMNS
from src.preprocessing.addresses import normalize_address
from src.preprocessing.names import normalize_name
from src.preprocessing.normalization import APOSTROPHES, canonicalize, compact, fold_latin_marks
from src.preprocessing.records import NORMALIZED_ADDRESS, NORMALIZED_NAME, iter_normalized_source, normalize_record

SAMPLES = [
    '', ' ', 'ACME, Inc.', '  Madelina  Fleming\tHeartland ', 'राम', 'École', 'Straße', 'ＡＢＣ１２３',
    "O'Brien's", 'Tri-State', 'AT&T', 'Shreeraj &-Partners', 'NA', 'NULL', 'None', 'L.L.C.', '#1303 MONTOYA ST.',
    'Oppï¿½Sushma', 'Prívate Ínc', 'హార్డ్‌వేర్', 'ﬁnance ①', 'ŉ', 'क़िला', 'İstanbul', '½', '...', 'á',
    'விஜன் ட்ரேடிங்', 'Mumbai, महाराष्ट्र 400001', 'PEOPLEÂ\x80\x99S', 'São Paulo – Nº 5', '<NULL>', 'x‌́',
]


@pytest.mark.parametrize('raw,expected', [
    ('', ''),
    ('   \t\n ', ''),
    ('  Madelina  Fleming\tHeartland \n', 'madelina fleming heartland'),
    ('HAPPY HAIR STUDIO LP', 'happy hair studio lp'),
    ('Straße', 'strasse'),
    ('ACME, Inc.', 'acme inc'),
    ('Tri-State Center', 'tri state center'),
    ('L.L.C.', 'l l c'),
    ('*** Sai Tech | www.sai.com', 'sai tech www sai com'),
    ('>> Prime [Consultants]', 'prime consultants'),
    ('Shreeraj &-Partners', 'shreeraj & partners'),
    ('AT&T', 'at & t'),
    ("O'Brien's", 'obriens'),
    ('GOVERNOR`S GATE', 'governors gate'),
    ('Candy’s', 'candys'),
    ('#1303 Montoya St.', '1303 montoya st'),
    ('T C 15/43 (1) & (2)', 't c 15 43 1 & 2'),
    ('015115 Brookview', '015115 brookview'),
    ('2Nd Floor, Phase-1', '2nd floor phase 1'),
    ('ＡＢＣ１２３', 'abc123'),
    ('ﬁnance ①', 'finance 1'),
    ('10 Main', '10 main'),
])
def test_canonical_examples(raw, expected):
    assert canonicalize(raw) == expected


@pytest.mark.parametrize('raw,expected', [
    ('NA', 'na'), ('N/A', 'n a'), ('NULL', 'null'), ('None', 'none'), ('nan', 'nan'), ('<NA>', 'na'), ('<NULL>', 'null'),
])
def test_literal_null_like_strings_remain_text(raw, expected):
    assert canonicalize(raw) == expected


def test_non_string_input_rejected():
    with pytest.raises(TypeError):
        canonicalize(None)


@pytest.mark.parametrize('raw,expected', [
    ('राम', 'राम'),
    ('विजन ट्रेडिंग', 'विजन ट्रेडिंग'),
    ('उत्तर प्रदेश', 'उत्तर प्रदेश'),
    ('अरिहंत। ट्रेडिंग', 'अरिहंत ट्रेडिंग'),
    ('அரிஹந்த் டிரேடிங்', 'அரிஹந்த் டிரேடிங்'),
    ('ಗೋಲ್ಡ್ ಎಸ್ಟೇಟ್', 'ಗೋಲ್ಡ್ ಎಸ್ಟೇಟ್'),
    ('École Normale', 'école normale'),
    ('Café Zürich', 'café zürich'),
    ('SÃO PAULO', 'são paulo'),
])
def test_unicode_scripts_and_marks_preserved(raw, expected):
    assert canonicalize(raw) == expected


def test_decomposed_and_composed_accents_agree():
    assert canonicalize('Café') == canonicalize('Café') == 'café'


def test_format_characters_deleted_not_split():
    assert canonicalize('హార్డ్‌వేర్') == 'హార్డ్వేర్'
    assert canonicalize('ab​cd­') == 'abcd'


def test_corruption_marker_does_not_invent_digits():
    assert canonicalize('Oppï¿½Sushma') == canonicalize('OPPÏ¿½SUSHMA') == 'opp sushma'
    assert canonicalize('A�B') == 'a b'
    assert canonicalize('½') == '1 2'


def test_numeric_information_preserved():
    for raw in ['Plot 372, 6Th Cross 560027', '1225 17, Russell', 'No : 5 Ground Floor', 'Unit #136']:
        digits = [c for c in raw if c.isdigit()]
        assert [c for c in canonicalize(raw) if c.isdigit()] == digits


@pytest.mark.parametrize('raw', SAMPLES)
def test_idempotent_deterministic_and_nonempty_text_retained(raw):
    once = canonicalize(raw)
    assert canonicalize(once) == once == canonicalize(raw)
    assert fold_latin_marks(fold_latin_marks(once)) == fold_latin_marks(once)
    assert compact(compact(once)) == compact(once)
    if any(unicodedata.category(c)[0] in 'LN' for c in raw):
        assert once
    assert '  ' not in once and once == once.strip()


def test_letters_and_digits_of_every_script_retained():
    for raw in SAMPLES:
        folded = unicodedata.normalize('NFKC', unicodedata.normalize('NFKC', raw).casefold())
        expected = [c for c in unicodedata.normalize('NFD', folded)
                    if unicodedata.category(c)[0] in 'LN' and c not in APOSTROPHES]
        got = [c for c in unicodedata.normalize('NFD', canonicalize(raw)) if unicodedata.category(c)[0] in 'LN']
        if 'ï¿½' not in raw and 'Ï¿½' not in raw:
            assert got == expected, raw


def test_no_semantic_substitutions():
    assert canonicalize('10 Market Street Road Avenue') == '10 market street road avenue'
    assert canonicalize('10 Market St Rd Ave') == '10 market st rd ave'
    assert canonicalize('ACME Pvt. Ltd. Private Limited Inc') == 'acme pvt ltd private limited inc'
    assert canonicalize('Smith and Sons') != canonicalize('Smith & Sons')
    assert canonicalize('015115') != canonicalize('15115')


def test_compact_view_is_separate():
    assert compact(canonicalize('Tri-State Center')) == 'tristatecenter'
    assert compact(canonicalize('L.L.C.')) == compact(canonicalize('LLC')) == 'llc'
    assert canonicalize('L.L.C.') != canonicalize('LLC')
    assert compact('') == ''


def test_fold_latin_marks_only_touches_latin():
    assert fold_latin_marks(canonicalize('Prívate Ínc École')) == 'private inc ecole'
    for text in ['राम', 'क़िला', 'उत्तर प्रदेश', 'அரிஹந்த்', 'ಗೋಲ್ಡ್']:
        assert fold_latin_marks(canonicalize(text)) == canonicalize(text)
    assert fold_latin_marks('ø ł æ strasse') == 'ø ł æ strasse'
    assert fold_latin_marks(canonicalize('İstanbul')) == 'istanbul'
    assert canonicalize('École') == 'école'


def test_name_and_address_apis_are_canonical():
    for raw in SAMPLES:
        assert normalize_name(raw) == normalize_address(raw) == canonicalize(raw)


def test_raw_record_unchanged(tmp_path):
    record = {'entity_id': 'S2-1', 'business_name': 'VISION  TRADING Pvt.', 'business_address': '', 'country': 'India'}
    original = copy.deepcopy(record)
    normalized = normalize_record(record)
    assert record == original
    assert {k: normalized[k] for k in SOURCE_COLUMNS} == original
    assert normalized[NORMALIZED_NAME] == 'vision trading pvt'
    assert normalized[NORMALIZED_ADDRESS] == ''
    path = tmp_path / 'source.tsv'
    rows = [('S3-1', 'विजन ट्रेडिंग', 'NULL', 'India'), ('S3-2', 'NA', '', 'France')]
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, delimiter='\t', lineterminator='\n')
        writer.writerows([SOURCE_COLUMNS, *rows])
    before = path.read_bytes()
    streamed = list(iter_normalized_source(path, 3))
    assert path.read_bytes() == before
    assert [tuple(r[c] for c in SOURCE_COLUMNS) for r in streamed] == rows
    assert [(r[NORMALIZED_NAME], r[NORMALIZED_ADDRESS]) for r in streamed] == [('विजन ट्रेडिंग', 'null'), ('na', '')]
