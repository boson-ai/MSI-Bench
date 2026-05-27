# Interaction-Bench

A benchmark and dataset scaffold for evaluating modern Audio Understanding Models in realistic assistant interactions.

## Repository Layout

```text
Interaction-Bench/
  README.md
  docs/
    taxonomy.md                    # scenario, action, perturbation taxonomy
    Taxonomy2.0.md                 # expanded taxonomy specification
    metrics.md                     # scoring definitions and prediction format
    data_construction_pipelines.md # data construction pipeline documentation
  schema/
    interaction_bench.schema.json  # JSON Schema for manifest validation
    scenario_templates.json        # controlled interaction script templates
  scripts/
    build_seed_manifest.py         # normalize local WearVox + HumDial-FDBench seeds
    validate_manifest.py           # schema-light validation with stdlib only
    score_predictions.py           # engagement/action + answer scoring
  examples/
    seed_manifest.jsonl            # generated seed examples
    oracle_predictions.jsonl       # oracle predictions for sanity checking
  reports/
    seed_summary.json              # generated manifest summary
    oracle_score.json              # oracle prediction scoring output
  tests/
    test_scripts.py                # unit tests for scripts
```

## Quick Start

From the workspace root:

```bash
python Interaction-Bench/scripts/build_seed_manifest.py \
  --wearvox WearVox \
  --humdial HumDial-FDBench/test \
  --output Interaction-Bench/examples/seed_manifest.jsonl \
  --summary Interaction-Bench/reports/seed_summary.json \
  --limit-per-source 50

python Interaction-Bench/scripts/validate_manifest.py \
  --check-audio \
  Interaction-Bench/examples/seed_manifest.jsonl
```

Optional scoring with model predictions:

```bash
python Interaction-Bench/scripts/score_predictions.py \
  --manifest Interaction-Bench/examples/seed_manifest.jsonl \
  --predictions predictions.jsonl \
  --output metrics.json
```

Each prediction line should contain at least:

```json
{"id": "wearvox-head-0", "predicted_action": "respond", "predicted_answer": "Madrid"}
```

For silence cases, use `"predicted_action": "ignore"` and an empty or omitted `predicted_answer`.

## Manifest Portability

The validator resolves relative audio paths from the current directory and the manifest's ancestors, so the same manifest can be checked from either the workspace root or this repository directory. Use `--audio-root` when validating a manifest stored outside the workspace tree.

## Development Checks

```bash
python -m unittest discover -s tests -v
python -m py_compile scripts/*.py tests/*.py
```

The oracle prediction fixture should score at 1.0 engagement accuracy and 1.0 exact match for answerable rows:

```bash
python scripts/score_predictions.py \
  --manifest examples/seed_manifest.jsonl \
  --predictions examples/oracle_predictions.jsonl \
  --output reports/oracle_score.json
```
