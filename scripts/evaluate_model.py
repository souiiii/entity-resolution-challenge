from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.validation import file_hash, read_config, search_universe, training_target_pool, verify_manifests
from src.evaluation.evaluate import EVALUATION_PROTOCOL, evaluate_predictions, run_sanity_checks
from src.utils.paths import PROJECT_ROOT, RAW_DIR, INTERIM_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description='Exact per-Source-1 Macro F0.5; micro diagnostics are not the challenge score')
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--truth', type=Path, default=RAW_DIR / 'train/train_ground_truth.tsv')
    parser.add_argument('--split-dir', type=Path, default=INTERIM_DIR / 'validation')
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'experiments/configs/validation.json')
    parser.add_argument('--partition', choices=('development', 'validation'), default='validation')
    parser.add_argument('--target-universe', choices=('full', 'partition-pool'), default='full',
                        help='full: complete training S2+S3 corpus (validation search universe); '
                             'partition-pool: ownership-isolated development training pool')
    parser.add_argument('--work-dir', type=Path, default=INTERIM_DIR)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--sanity-checks', action='store_true')
    args = parser.parse_args()
    if bool(args.predictions) == args.sanity_checks:
        parser.error('Provide exactly one of --predictions or --sanity-checks')
    if args.target_universe == 'partition-pool' and (args.partition != 'development' or args.sanity_checks):
        parser.error('partition-pool is a development training diagnostic; validation searches the full universe')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    metadata = verify_manifests(args.split_dir, read_config(args.config))
    if file_hash(args.truth) != metadata['input_hashes']['train_ground_truth.tsv']:
        raise ValueError('Ground truth differs from frozen split input')
    if args.sanity_checks:
        if args.partition != 'validation':
            parser.error('--sanity-checks uses validation only')
        result = run_sanity_checks(args.truth, args.split_dir, args.work_dir)
    else:
        universe = (search_universe(args.split_dir) if args.target_universe == 'full'
                    else training_target_pool(args.split_dir, args.partition))
        result = evaluate_predictions(args.truth, args.predictions,
            args.split_dir / f'{args.partition}_source1.tsv', universe, args.work_dir)
    report = {'split_fingerprint': metadata['fingerprint'], 'evaluation_protocol': EVALUATION_PROTOCOL,
              'partition': args.partition, 'target_universe': args.target_universe, 'results': result}
    content = json.dumps(report, indent=2, sort_keys=True) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)
    print(content, end='')


if __name__ == '__main__':
    main()
