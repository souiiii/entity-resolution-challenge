from __future__ import annotations

import json

from src.data.inspect import summarize_table
from src.data.load import list_input_files
from src.utils.paths import RAW_DIR


def main() -> None:
    files = list_input_files(RAW_DIR)
    if not files:
        print(f"No CSV/TSV files found under: {RAW_DIR}")
        print("Copy official files into data/raw/train and data/raw/test.")
        return

    for path in files:
        print("=" * 80)
        print(json.dumps(summarize_table(path), indent=2))


if __name__ == "__main__":
    main()
