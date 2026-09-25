# Business Entity Resolution Challenge

Starter repository for the challenge.

## Problem

Given three datasets of business records, identify which Source 2 and Source 3 records correspond to each Source 1 business.

A Source 1 entity may have no matches, one match, or multiple matches. The scoring metric is Macro F0.5, so precision matters more than recall.

## First Steps

1. Put official files into:

```text
data/raw/train/
data/raw/test/
```

2. Create an environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Inspect data:

```bash
python scripts/inspect_data.py
```

4. Run tests:

```bash
pytest
```

5. Prepare the frozen validation split and verify the scorer:

```bash
python scripts/prepare_validation_split.py
python scripts/evaluate_model.py --sanity-checks
```

Later experiments use `python scripts/evaluate_model.py --predictions <path.tsv>`.
Read [VALIDATION_PROTOCOL.md](VALIDATION_PROTOCOL.md) for the fixed split,
leakage rules, exact metric, and reproduction commands.

Read `AGENTS.md` before asking Codex to implement phases.
