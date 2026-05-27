# Interaction-Bench

A benchmark and dataset scaffold for evaluating modern Audio Understanding Models in realistic assistant interactions where the output is a **text answer or a deliberate silence decision**, not a transcript.

## Objective

Interaction-Bench measures whether an audio model can behave like a socially aware assistant:

- answer when the user or an accepted participant addresses the assistant;
- stay silent when speech is part of a side conversation;
- wait when the conversational turn is incomplete or interrupted;
- incorporate a newly joined participant who explicitly addresses the assistant;
- remain robust under real-world acoustic perturbations such as overlap, reverb, wind, car noise, honks, TV, bystanders, and codec degradation.

## First-version non-goals

- No model training or fine-tuning.
- No transcript benchmark as the primary objective. Transcripts can appear as metadata, but scoring targets interaction behavior and answer quality.

## Repository layout

```text
Interaction-Bench/
  README.md
  docs/
    taxonomy.md              # scenario, action, perturbation taxonomy
    metrics.md               # scoring definitions and prediction format
  schema/
    interaction_bench.schema.json
  scripts/
    build_seed_manifest.py   # normalize local WearVox + HumDial-FDBench seeds
    validate_manifest.py     # schema-light validation with stdlib only
    score_predictions.py     # engagement/action + answer scoring
  examples/
    seed_manifest.jsonl      # generated seed examples
  reports/
    seed_summary.json        # generated manifest summary
```

## Quick start

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
{"id":"wearvox-head-0","predicted_action":"respond","predicted_answer":"Madrid"}
```

For silence cases, use `"predicted_action":"ignore"` and an empty or omitted `predicted_answer`.

## Manifest portability

The validator resolves relative audio paths from the current directory and the manifest's ancestors, so the same manifest can be checked from either the workspace root or this repository directory. Use `--audio-root` when validating a manifest stored outside the workspace tree.

## Development checks

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

## Related seed sources

The scaffold currently normalizes only real-data local copies from the existing-work discussion:

- **WearVox**: use only `task == "non-assistant-directed"` side-talk rejection rows. QA, tool-calling, and translation rows are excluded from the taxonomy seed set. WearVox scenes are mapped to Axis A per row from metadata/transcript rather than to a generic wearable class; for example, outdoor rows seed Public / Civic Space, while indoor office-like rows seed Work / Professional.
- **HumDial-FDBench**: use real full-duplex test folders for participation and turn-taking labels, including assistant-directed turns, wait/backchannel/pause cases, and side conversations. HumDial folder names primarily support Axis B; Axis A scene labels require transcript-level audit.

These are treated as seed sources. The benchmark taxonomy is broader than either dataset and is designed to support future newly recorded examples without mixing synthetic data into the current real-data seed mapping.
