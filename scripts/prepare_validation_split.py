from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.validation import prepare_split, read_config
from src.utils.paths import PROJECT_ROOT, RAW_DIR, INTERIM_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare or verify the frozen leakage-safe validation split')
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'experiments/configs/validation.json')
    parser.add_argument('--train-dir', type=Path, default=RAW_DIR / 'train')
    parser.add_argument('--output-dir', type=Path, default=INTERIM_DIR / 'validation')
    parser.add_argument('--rebuild', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    metadata = prepare_split(args.train_dir, args.output_dir, read_config(args.config), args.rebuild)
    print(json.dumps({'fingerprint': metadata['fingerprint'], 'partitions': metadata['partitions'],
                      'integrity_checks': metadata['integrity_checks']}, indent=2))


if __name__ == '__main__':
    main()
