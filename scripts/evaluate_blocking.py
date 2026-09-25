from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.blocking.channels import load_config
from src.blocking.evaluate import run_blocking
from src.utils.paths import INTERIM_DIR, PROJECT_ROOT, RAW_DIR

BLOCKING_DIR = INTERIM_DIR / 'blocking'


def summary(report: dict) -> str:
    recall = report['recall']
    lines = [f"{report['blocking_version']} mode={report['mode']} queries={report['queries']:,} "
             f"links={report['links']} runtime={report['resources']['runtime_seconds']}s "
             f"peak_rss={report['resources']['peak_rss_mib']}MiB",
             'recall: ' + ', '.join(f"{k} {v['recall']:.4f} ({v['found']:,}/{v['links']:,})"
                                    for k, v in recall.items() if k != 'by_country'),
             'recall by country: ' + ', '.join(f"{c} {v['overall']['recall']:.4f}" for c, v in recall['by_country'].items())]
    for name in ('source2', 'source3', 'combined'):
        d = report['candidates'][name]
        lines.append(f"candidates {name}: pairs {d['total_pairs']:,} mean {d['mean']:.1f} p50 {d['p50']} p90 {d['p90']} "
                     f"p95 {d['p95']} p99 {d['p99']} max {d['max']} >100 {d['share_over_100']:.4f} >500 {d['share_over_500']:.4f}")
    e = report['entities']
    lines.append(f"entities: all links {e['all_links_recovered']:,}/{e['with_links']:,}, none {e['zero_recovered']:,}, "
                 f"singleton median candidates {e['singleton_candidates'].get('p50')}")
    for channel in report['channels']:
        cells = []
        for s in ('source2', 'source3'):
            c = channel.get(s)
            if c:
                cells.append(f"{s[-2:]} alone {c['standalone_recall']:.4f} +true {c['incremental_true']:,} "
                             f"+pairs {c['incremental_candidates']:,} capped {c['queries_capped']:,}")
        lines.append(f"  {channel['name']}: " + ' | '.join(cells))
    for s, m in report['missed'].items():
        lines.append(f"missed {s}: {m['missed_links']:,} no shared key {m['no_shared_key_in_any_channel']:,} "
                     f"only above limits {m['shared_key_only_above_limits']:,}")
    lines.append(f"stores {sum(report['resources']['store_bytes'].values()) / 2**20:.0f} MiB, "
                 f"indexes {report['resources']['index_bytes']}, disk free after {report['resources']['disk_free_after_gib']} GiB")
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description='Development-only deterministic blocking evaluation (no matching)')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--sample', action='store_true', help='hash-selected development sample')
    mode.add_argument('--full', action='store_true', help='all development Source 1 entities')
    mode.add_argument('--clean', action='store_true', help='delete the derived key stores and indexes and exit')
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'experiments/configs/blocking_v1.json')
    parser.add_argument('--validation-config', type=Path, default=PROJECT_ROOT / 'experiments/configs/validation.json')
    parser.add_argument('--raw-dir', type=Path, default=RAW_DIR / 'train')
    parser.add_argument('--split-dir', type=Path, default=INTERIM_DIR / 'validation')
    parser.add_argument('--store-dir', type=Path, default=BLOCKING_DIR / 'store')
    parser.add_argument('--index-dir', type=Path, default=BLOCKING_DIR / 'index')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--write-sample-candidates', action='store_true',
                        help='with --sample, also write a debug candidate file in the official format')
    parser.add_argument('--sample-candidate-rows', type=int, default=2_000,
                        help='number of sample Source 1 rows written by --write-sample-candidates')
    args = parser.parse_args()
    if args.clean:
        for directory in (args.store_dir, args.index_dir):
            shutil.rmtree(directory, ignore_errors=True)
            print(f'removed {directory}')
        return
    if args.write_sample_candidates and not args.sample:
        parser.error('--write-sample-candidates requires --sample')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    config = load_config(args.config)
    name = 'sample' if args.sample else 'full'
    output = args.output or BLOCKING_DIR / f'{config.version}_{name}_report.json'
    candidates = BLOCKING_DIR / f'{config.version}_sample_candidate_pairs.tsv' if args.write_sample_candidates else None
    report = run_blocking(config, args.raw_dir, args.split_dir, args.validation_config, args.store_dir,
                          args.index_dir, sample=args.sample, sample_candidates_path=candidates,
                          sample_candidate_rows=args.sample_candidate_rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(summary(report))
    print(f'report: {output}')


if __name__ == '__main__':
    main()
