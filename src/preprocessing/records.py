from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path

from src.data.load import iter_source
from src.preprocessing.addresses import normalize_address
from src.preprocessing.names import normalize_name

NORMALIZED_NAME = 'business_name_norm'
NORMALIZED_ADDRESS = 'business_address_norm'


def normalize_record(record: Mapping[str, str]) -> dict[str, str]:
    """Return a new record with canonical fields added beside the untouched raw fields."""
    return {**record, NORMALIZED_NAME: normalize_name(record['business_name']),
            NORMALIZED_ADDRESS: normalize_address(record['business_address'])}


def iter_normalized_source(path: str | Path, source: int) -> Iterator[dict[str, str]]:
    """Stream a source with raw and canonical name/address fields side by side."""
    for record in iter_source(path, source):
        yield normalize_record(record)
