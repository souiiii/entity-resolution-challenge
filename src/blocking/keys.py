"""Deterministic blocking keys derived from the Phase 2 canonical representations.

Every key is a 64-bit BLAKE2b hash that includes the record's raw country label, so
equal keys can only occur within one country and unseen labels need no special
handling. Zero is reserved for an empty value.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from src.preprocessing.normalization import canonicalize, compact, fold_latin_marks

EXACT_FIELDS = ('name_canonical', 'name_folded', 'name_folded_compact', 'address_canonical')
WORD, NUMBER = 0, 1


def key_hash(kind: str, country: str, value: str) -> int:
    """Unsigned 64-bit key for one (kind, country, value); 0 only for an empty value."""
    if not value:
        return 0
    digest = hashlib.blake2b(f'{kind}\x1f{country}\x1f{value}'.encode('utf-8'), digest_size=8).digest()
    return int.from_bytes(digest, 'little') or 1


def number_form(token: str) -> str:
    """Integer-equivalent form of an all-digit token (``015115`` -> ``15115``)."""
    return str(int(token))


@dataclass(frozen=True)
class RecordKeys:
    exact: tuple[int, int, int, int]
    name_tokens: tuple[int, ...]
    address_tokens: tuple[int, ...]
    address_kinds: tuple[int, ...]


def record_keys(name: str, address: str, country: str) -> RecordKeys:
    """Exact keys, unique folded name tokens and ordered address word/number tokens."""
    name_canonical = canonicalize(name)
    name_folded = fold_latin_marks(name_canonical)
    address_canonical = canonicalize(address)
    exact = (key_hash('name_canonical', country, name_canonical),
             key_hash('name_folded', country, name_folded),
             key_hash('name_folded_compact', country, compact(name_folded)),
             key_hash('address_canonical', country, address_canonical))
    name_tokens = tuple(key_hash('name_token', country, token) for token in dict.fromkeys(name_folded.split()))
    address_tokens, address_kinds = [], []
    for token in address_canonical.split():
        if token.isdecimal():
            address_tokens.append(key_hash('address_number', country, number_form(token)))
            address_kinds.append(NUMBER)
        else:
            address_tokens.append(key_hash('address_word', country, token))
            address_kinds.append(WORD)
    return RecordKeys(exact, name_tokens, tuple(address_tokens), tuple(address_kinds))
