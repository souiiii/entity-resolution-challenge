from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.blocking.channels import estimate_postings, load_config
from src.blocking.store import RecordStore
from src.utils.paths import INTERIM_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description='Read-only posting and disk bounds before index construction')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--blocking-dir', type=Path, default=INTERIM_DIR / 'blocking')
    args = parser.parse_args()
    config = load_config(args.config)
    result = {'free_gib': shutil.disk_usage(args.blocking_dir).free / 2**30,
              'reserve_gib': config.min_free_disk_gib, 'indexes': []}
    for source in (2, 3):
        store = RecordStore(args.blocking_dir / 'store' / f'train_source{source}')
        for spec in config.channels:
            if source not in spec.sources:
                continue
            directory = args.blocking_dir / 'index' / f'source{source}' / spec.index_id
            meta_path = directory / 'meta.json'
            cached = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            generated = estimate_postings(spec, store)
            result['indexes'].append({'source': source, 'channel': spec.name, 'index_id': spec.index_id,
                                      'cached_bytes': cached.get('bytes'), 'generated_postings_upper_bound': generated,
                                      'persistent_upper_gib': generated * 12 / 2**30,
                                      'build_upper_gib': generated * 24 / 2**30})
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
