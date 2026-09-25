"""Streaming writers for the official ID-list submission files."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

MATCHING_HEADER = ('source1_entity_id', 'matched_entity_ids')
CANDIDATE_HEADER = ('source1_entity_id', 'candidate_entity_ids')
_HEADERS = (MATCHING_HEADER, CANDIDATE_HEADER)


def write_id_list_file(path: str | Path, header: Sequence[str], rows: Iterable[tuple[str, Sequence[str]]]) -> int:
    """Write ``source1_id<TAB>id,id,...`` rows under an official header; returns the row count.

    Rows are written in the given order. Every Source 1 ID must be unique, every
    list item an ``S2-``/``S3-`` ID without repeats, and an empty list is written as
    an empty field. The file is written to a temporary path and renamed on success.
    """
    header = tuple(header)
    if header not in _HEADERS:
        raise ValueError(f'Unsupported header {header}')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.partial')
    seen: set[str] = set()
    try:
        with partial.open('w', encoding='utf-8', newline='\n') as stream:
            stream.write('\t'.join(header) + '\n')
            for source1_id, targets in rows:
                if not source1_id.startswith('S1-') or any(c in source1_id for c in '\t\n,'):
                    raise ValueError(f'Invalid Source 1 ID {source1_id!r}')
                if source1_id in seen:
                    raise ValueError(f'Duplicate Source 1 row {source1_id}')
                seen.add(source1_id)
                targets = list(targets)
                if len(targets) != len(set(targets)):
                    raise ValueError(f'Repeated target for {source1_id}')
                for target in targets:
                    if not target.startswith(('S2-', 'S3-')) or any(c in target for c in '\t\n, '):
                        raise ValueError(f'Invalid target ID {target!r} for {source1_id}')
                stream.write(f"{source1_id}\t{','.join(targets)}\n")
        partial.replace(path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return len(seen)
