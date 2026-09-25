"""Reproduce Phase 1 statistics without loading full datasets into memory."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

# Support the documented direct invocation from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.inspect import inspect_challenge
from src.utils.paths import RAW_DIR, INTERIM_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, default=RAW_DIR)
    parser.add_argument('--work-dir', type=Path, default=INTERIM_DIR)
    parser.add_argument('--output', type=Path, default=INTERIM_DIR / 'data_profile.json')
    parser.add_argument('--inventory-only', action='store_true', help='Only print file sizes and headers')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    for path in sorted(args.raw_dir.rglob('*.tsv')):
        with path.open(encoding='utf-8', newline='') as stream:
            header = next(csv.reader(stream, delimiter='\t'))
        print(f'{path.name}: {path.stat().st_size:,} bytes; {header}', flush=True)
    if args.inventory_only:
        return
    report = inspect_challenge(args.raw_dir, args.work_dir, args.output)
    for name, summary in report['sources'].items():
        print(f"{name}: {summary['rows']:,} rows; {summary['duplicate_ids']:,} duplicate IDs")
    truth = report['ground_truth']
    print('Ground truth match counts: ' + json.dumps(truth['match_count_distribution']))
    print(f'Full statistics: {args.output}')


if __name__ == '__main__':
    main()
