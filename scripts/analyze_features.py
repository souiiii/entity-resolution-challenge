from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.analysis import run_feature_analysis, write_report
from src.utils.paths import INTERIM_DIR, PROJECT_ROOT, RAW_DIR

HEADLINE = ('blocking_rank', 'blocking_aggregate_rarity', 'blocking_support_families', 'name_folded_equal',
            'name_char_similarity', 'name_token_set_similarity', 'name_token_jaccard', 'name_script_mismatch',
            'address_canonical_equal', 'address_char_similarity', 'address_token_set_similarity',
            'address_token_jaccard', 'number_jaccard', 'number_set_equal', 'target_address_empty')


def summary(report: dict) -> str:
    p = report['performance']
    lines = [f"{report['features_version']} queries={report['queries']:,} pairs={p['pairs']:,} "
             f"setup={p['setup_seconds']}s streaming={p['streaming_wall_seconds']}s pairs/s={p['pairs_per_streaming_second']:,} "
             f"feature_cpu={p['feature_cpu_seconds']}s peak_rss={p['peak_rss_mib_main']}MiB/"
             f"{p['peak_rss_mib_largest_worker']}MiB",
             'counts: ' + ', '.join(f"{g} {c['positive']:,}+/{c['negative']:,}-" for g, c in report['counts'].items()),
             f"projection full development: {p['projected_full_development_wall_hours']} h wall, "
             f"{p['projected_full_development_feature_cpu_hours']} feature CPU h, "
             f"matrix {p['projected_full_development_matrix_gib']} GiB if materialized"]
    groups = [g for g in report['groups'] if ':' not in g or g.startswith('country:')]
    lines.append('AUC ' + ' | '.join(groups))
    for name in HEADLINE:
        cells = []
        for g in groups:
            a = report['groups'][g][name]['auc']
            cells.append('  -  ' if a is None else f'{a:.3f}')
        lines.append(f'  {name:32s} ' + ' | '.join(cells))
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description='Development-sample pairwise feature diagnostics (no model)')
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'experiments/configs/features_v1.json')
    parser.add_argument('--validation-config', type=Path, default=PROJECT_ROOT / 'experiments/configs/validation.json')
    parser.add_argument('--limit', type=int, help='smoke run on the first N sample queries')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    config = json.loads(args.config.read_text())
    report = run_feature_analysis(config, RAW_DIR / 'train', INTERIM_DIR / 'validation', args.validation_config,
                                  INTERIM_DIR / 'blocking' / 'store', INTERIM_DIR / 'blocking' / 'index',
                                  INTERIM_DIR / 'features' / 'text', PROJECT_ROOT, args.limit)
    name = f"{config['version']}_{'smoke' if args.limit else 'sample'}_report.json"
    output = args.output or INTERIM_DIR / 'features' / name
    write_report(report, output)
    print(summary(report))
    print(f'report: {output}')


if __name__ == '__main__':
    main()
