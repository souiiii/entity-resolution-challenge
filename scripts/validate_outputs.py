"""Memory-bounded validation of both official output files with the official validator's own checks.

The official command holds every candidate ID of a full-size ``candidate_pairs.tsv`` in
Python sets, which does not fit on small machines. This script applies the official
``validate_id_list_file`` to contiguous row shards of each file (with the full
test Source 2/3 ID universe, i.e. ``--check-ids``), and streams both files once to
check global row order, uniqueness, complete Source 1 coverage and that every
matching row is a subset of its candidate row.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.paths import INTERIM_DIR, OUTPUT_DIR, PROJECT_ROOT, RAW_DIR


def official_validator():
    spec = importlib.util.spec_from_file_location('official_validator',
                                                  PROJECT_ROOT / 'official/utils/validate_submission.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_outputs(matching: Path, candidate: Path, test_dir: Path, shard_rows: int, work_dir: Path) -> dict:
    """Return official shard errors and global invariant failures for both files."""
    validator = official_validator()
    warnings: list[str] = []
    required = validator.read_ids(str(test_dir / 'test_source1.tsv'))
    valid_ids = validator.load_match_targets(str(test_dir), warnings)
    errors: list[str] = []
    problems = {'rows': 0, 'order_or_duplicate': 0, 'row_mismatch': 0, 'not_subset': 0, 'unknown_source1': 0}
    work_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = Path(tempfile.mkdtemp(prefix='output-shards-', dir=work_dir))
    try:
        with matching.open(encoding='utf-8') as m_stream, candidate.open(encoding='utf-8') as c_stream:
            headers = (m_stream.readline(), c_stream.readline())
            previous, shard, shard_ids = None, 0, set()
            m_out = c_out = None
            seen = 0

            def flush() -> None:
                if m_out is None:
                    return
                m_out.close()
                c_out.close()
                for path, header, label in ((shard_dir / 'm.tsv', validator.MATCHING_HEADER, 'matched_entity_ids'),
                                            (shard_dir / 'c.tsv', validator.CANDIDATE_HEADER, 'candidate_entity_ids')):
                    validator.validate_id_list_file(str(path), header, label, shard_ids, valid_ids, errors)

            for m_line, c_line in zip(m_stream, c_stream):
                if seen % shard_rows == 0:
                    flush()
                    shard_ids = set()
                    m_out = (shard_dir / 'm.tsv').open('w', encoding='utf-8')
                    c_out = (shard_dir / 'c.tsv').open('w', encoding='utf-8')
                    m_out.write(headers[0])
                    c_out.write(headers[1])
                    shard += 1
                m_out.write(m_line)
                c_out.write(c_line)
                seen += 1
                m_id, _, m_rest = m_line.rstrip('\n').partition('\t')
                c_id, _, c_rest = c_line.rstrip('\n').partition('\t')
                shard_ids.add(m_id)
                problems['row_mismatch'] += m_id != c_id
                problems['order_or_duplicate'] += previous is not None and m_id <= previous
                problems['unknown_source1'] += m_id not in required
                previous = m_id
                matched = set(filter(None, m_rest.split(',')))
                problems['not_subset'] += not matched <= set(filter(None, c_rest.split(',')))
            flush()
            problems['rows'] = seen
            problems['extra_matching_lines'] = sum(1 for _ in m_stream)
            problems['extra_candidate_lines'] = sum(1 for _ in c_stream)
    finally:
        shutil.rmtree(shard_dir, ignore_errors=True)
    failures = {k: v for k, v in problems.items() if k != 'rows' and v}
    if problems['rows'] != len(required):
        failures['missing_source1_rows'] = len(required) - problems['rows']
    return {'status': 'PASS' if not errors and not failures else 'FAIL', 'official_shard_errors': errors,
            'global_failures': failures, 'rows': problems['rows'], 'required_source1': len(required),
            'valid_target_ids': len(valid_ids) if valid_ids is not None else None, 'shards': shard,
            'warnings': warnings}


def main() -> None:
    parser = argparse.ArgumentParser(description='Memory-bounded official validation of both submission files')
    parser.add_argument('--matching', type=Path, default=OUTPUT_DIR / 'matching_results.tsv')
    parser.add_argument('--candidate', type=Path, default=OUTPUT_DIR / 'candidate_pairs.tsv')
    parser.add_argument('--test-dir', type=Path, default=RAW_DIR / 'test')
    parser.add_argument('--shard-rows', type=int, default=200_000)
    args = parser.parse_args()
    result = validate_outputs(args.matching, args.candidate, args.test_dir, args.shard_rows, INTERIM_DIR / 'matching')
    print(json.dumps(result, indent=2))
    sys.exit(0 if result['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
