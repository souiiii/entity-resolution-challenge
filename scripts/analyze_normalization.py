from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.validation import read_config
from src.preprocessing.analysis import KEYS, REPRESENTATIONS, analyze_normalization
from src.utils.paths import INTERIM_DIR, PROJECT_ROOT, RAW_DIR


def summary(report: dict) -> str:
    lines = [f"normalization={report['normalization_version']} split={report['split_fingerprint'][:12]} "
             f"partition={report['partition']} runtime={report['runtime_seconds']}s",
             f"development links: {report['development_link_counts']}"]
    for source, result in report['link_equality'].items():
        total = result['total']
        lines.append(f"{source} positive-link equality (n={total['links']:,}):")
        for key in KEYS:
            cells = ', '.join(f"{rep} {total[key][rep]['count']:,} ({total[key][rep]['fraction']:.4%})"
                              for rep in REPRESENTATIONS)
            lines.append(f'  {key}: {cells}')
    lines.append('same-country equal-key cross pairs (true-link fraction):')
    for key in KEYS:
        for rep in REPRESENTATIONS:
            entry = report['collisions'][key][rep]
            cells = ', '.join(f"{name.split('_')[2]} {entry[name]['equal_key_pairs']:,} "
                              f"({entry[name]['true_link_fraction']:.4f})"
                              for name in ('source1_to_source2_same_country', 'source1_to_source3_same_country'))
            lines.append(f'  {key}/{rep}: {cells}')
    for source, data in report['sources'].items():
        for field, flags in data['flags'].items():
            lines.append(f"{source} {field}: empty_after={flags.get('empty_after_normalization', 0)} "
                         f"non_ascii_empty_after={flags.get('non_ascii_empty_after_normalization', 0)} "
                         f"lost={flags.get('letters_or_digits_lost', 0)} "
                         f"not_idempotent={flags.get('canonical_not_idempotent', 0)} "
                         f"devanagari_altered={flags.get('devanagari_altered', 0)} "
                         f"folding_altered_indic={flags.get('folding_altered_indic', 0)}")
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description='Development-only normalization diagnostics; no matching is performed')
    parser.add_argument('--raw-dir', type=Path, default=RAW_DIR / 'train')
    parser.add_argument('--split-dir', type=Path, default=INTERIM_DIR / 'validation')
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'experiments/configs/validation.json')
    parser.add_argument('--work-dir', type=Path, default=INTERIM_DIR)
    parser.add_argument('--output', type=Path, default=INTERIM_DIR / 'normalization_analysis.json')
    parser.add_argument('--workers', type=int)
    parser.add_argument('--no-examples', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    report = analyze_normalization(args.raw_dir, args.split_dir, read_config(args.config), args.work_dir,
                                   include_examples=not args.no_examples, workers=args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + '\n')
    print(summary(report))
    print(f'full report: {args.output}')


if __name__ == '__main__':
    main()
