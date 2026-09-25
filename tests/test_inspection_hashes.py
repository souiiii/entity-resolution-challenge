from pathlib import Path
import subprocess
import sys

import pytest

from src.data.inspect import verify_input_hashes


def test_hash_workflow_and_tamper_detection(tmp_path):
    raw, official = tmp_path / 'raw', tmp_path / 'official'
    raw.mkdir()
    official.mkdir()
    (raw / 'sample.tsv').write_text('id\nS1-1\n')
    (official / 'README.md').write_text('official')
    manifest = tmp_path / 'interim/hashes.json'
    script = Path(__file__).resolve().parents[1] / 'scripts/inspect_data.py'
    command = [sys.executable, str(script), '--hashes-only', '--raw-dir', str(raw),
               '--official-dir', str(official), '--hash-manifest', str(manifest)]
    subprocess.run(command, check=True, capture_output=True)
    original = manifest.read_bytes()
    subprocess.run(command, check=True, capture_output=True)
    assert manifest.read_bytes() == original
    (raw / 'sample.tsv').write_text('changed')
    with pytest.raises(ValueError, match='hashes differ'):
        verify_input_hashes(raw, official, manifest)
    assert manifest.read_bytes() == original


def test_default_inspection_creates_and_verifies_hashes(tmp_path):
    import json
    raw, official, interim = (tmp_path / name for name in ('raw', 'official', 'interim'))
    official.mkdir()
    (official / 'README.md').write_text('official fixture')
    for split, suffix in [('train', 'a'), ('test', 't')]:
        (raw / split).mkdir(parents=True)
        for source in (1, 2, 3):
            (raw / split / f'{split}_source{source}.tsv').write_text(
                f'entity_id\tbusiness_name\tbusiness_address\tcountry\nS{source}-{suffix}\tName\tAddress\tUS\n')
    (raw / 'train/train_ground_truth.tsv').write_text('source1_entity_id\tmatched_entity_ids\nS1-a\tS2-a,S3-a\n')
    script = Path(__file__).resolve().parents[1] / 'scripts/inspect_data.py'
    command = [sys.executable, str(script), '--raw-dir', str(raw), '--official-dir', str(official),
               '--work-dir', str(interim), '--hash-manifest', str(interim / 'hashes.json'),
               '--output', str(interim / 'profile.json')]
    subprocess.run(command, check=True, capture_output=True)
    assert json.loads((interim / 'profile.json').read_text())['complete']
    hashes = json.loads((interim / 'hashes.json').read_text())
    assert len(hashes) == 8
    assert verify_input_hashes(raw, official, interim / 'hashes.json') == hashes
