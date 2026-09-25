"""Conservative, Unicode-safe canonical text normalization shared by names and addresses."""
from __future__ import annotations

import re
import unicodedata

NORMALIZATION_VERSION = 'canonical-v1'
KEPT_SYMBOLS = frozenset('&')
APOSTROPHES = frozenset("'`\u00b4\u2018\u2019\u201b\u02bc\uff07\uff40")
_CORRUPTION_MARKERS = re.compile('\ufffd|[\u00ef\u00cf]\u00bf\u00bd')
_APOSTROPHES = re.compile(f"[{''.join(sorted(APOSTROPHES))}]")


def canonicalize(text: str) -> str:
    """Return the canonical form of one raw field.

    Steps: replace U+FFFD and its Latin-1 mojibake with a separator; delete
    apostrophe-like elision marks; NFKC, Unicode case folding, NFKC; keep letters,
    combining marks and digits of every script; keep ``&`` as its own token; delete
    invisible format characters (Cf); turn every other character into a separator;
    compose (NFKC) and collapse separators to single spaces. No word is added,
    removed or substituted. The empty string maps to itself and literal strings
    such as ``NA`` or ``NULL`` stay ordinary text.
    """
    if not isinstance(text, str):
        raise TypeError(f'Expected str, got {type(text).__name__}')
    text = _APOSTROPHES.sub('', _CORRUPTION_MARKERS.sub(' ', text))
    text = _APOSTROPHES.sub('', unicodedata.normalize('NFKC', unicodedata.normalize('NFKC', text).casefold()))
    pieces = []
    for char in text:
        category = unicodedata.category(char)
        if category[0] in 'LMN':
            pieces.append(char)
        elif char in KEPT_SYMBOLS:
            pieces.append(f' {char} ')
        elif category != 'Cf':
            pieces.append(' ')
    return ' '.join(unicodedata.normalize('NFKC', ''.join(pieces)).split())


def compact(normalized: str) -> str:
    """Auxiliary view: an already normalized string with its token spaces removed."""
    return normalized.replace(' ', '')


def _is_latin(char: str) -> bool:
    return unicodedata.name(char, '').startswith('LATIN ')


def fold_latin_marks(normalized: str) -> str:
    """Auxiliary lossy view: drop nonspacing marks attached to Latin letters only.

    ``école`` becomes ``ecole``; marks on Devanagari and every other script, and
    Latin letters without a decomposition (``ø``, ``ł``, ``æ``), are unchanged.
    """
    kept, base_is_latin = [], False
    for char in unicodedata.normalize('NFD', normalized):
        if unicodedata.category(char) == 'Mn':
            if base_is_latin:
                continue
        else:
            base_is_latin = _is_latin(char)
        kept.append(char)
    return unicodedata.normalize('NFC', ''.join(kept))
