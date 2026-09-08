# MSI-Bench — evaluation harness

Evaluation code for **MSI-Bench** (Multi-Speaker Interaction Benchmark): the
prediction runners, deterministic scoring, LLM answer judge, and the table
aggregation used in the paper *MSI-Bench: Evaluating Multi-Speaker Voice
Interaction for Collaborative AI Agents* (arXiv: TODO). The dataset lives at
DATASET_URL_TODO; the dataset-construction pipeline is not part of this
repository.

## Layout

- `src/ib/` — the `ib` CLI: `eval`, `answer-judge`, `manifest-eval`,
  `rubric-eval`, `validate-predictions`, `score-smoke`
- `configs/` — eval target definitions (`--models`)
- `serve/` — vLLM serving scripts + configs for open-weight audio models
- `slurm/` — batch harness (`slurm/submit.sh <model-key> --suite <dir>`) and the
  per-model target files used for the paper (`slurm/models.d/*.models.yaml`)
- `tools/` — release-package adapter and paper-table aggregation
- `tests/` — unit tests for the shipped components (`uv run pytest`)

## Quick start

```bash
uv sync                       # or: pip install -e .
cp .env.example .env          # fill in the provider keys you need
python tools/build_suite_from_release.py --release <msi-bench-v1 dir> --output suite/ --probes
ib eval --run suite/ --models configs/eval-models.yaml --output artifacts/eval/run1 \
    --answer-judge --judge-provider openrouter --judge-model deepseek/deepseek-v4-pro
python tools/paper_table_rows.py --suite suite/ --runs artifacts/eval/run1
```

`build_suite_from_release.py` turns the release package (`data/…/case.json` +
`index/cases.jsonl`, plus `probes/*.jsonl` with `--probes`) into a suite: one
`<entry>/manifest/manifest.jsonl` per (language, pattern) with the referenced
audio. `ib eval` writes one target directory per model with
`predictions.jsonl`, `manifest_eval.json` (tool validation + probe summaries)
and `answer_judge.json`; `paper_table_rows.py` folds those into the paper's
columns.

## Reproducing the paper setting

- **Judge:** DeepSeek V4 Pro through OpenRouter, JSON mode, provider default
  temperature (`--answer-judge --judge-provider openrouter --judge-model
  deepseek/deepseek-v4-pro`, `OPENROUTER_API_KEY` in `.env`).
- **Open-weight models:** served with vLLM (`serve/serve_model.sh`, configs
  under `serve/configs/`), evaluated through the `local` provider with
  schema-constrained decoding: `export IB_LOCAL_GUIDED_JSON=1` before
  `ib eval`. Thinking variants (Gemma 4, MiMo-Audio) are declared in the
  matching `slurm/models.d/*.models.yaml`.
- **Hosted models:** `openai` (GPT Audio), `openai_realtime` (GPT Realtime),
  `gemini` providers; see `configs/eval-models.yaml` and `slurm/models.d/`.
- **Probes:** build the suite with `--probes`. Hear-time probes (PRR) run on
  every target (`--case-modes base,hear`). Speak-time probes (BIR) run natively
  on full-duplex targets with `--live-probe`, and on turn-based targets through
  the barge-in approximation `--speak-probe-approx`.
- **Metrics:** `tools/paper_table_rows.py` prints APR (with 95% Wilson
  half-width), ARS, Tool, BIR, PRR and per-pattern APR per target, with the
  paper's policies (format errors fail every atom; PRR excludes format-error
  rows; Tool over cases requiring a call).

## Output format expected from a model

One JSON object per case:

```json
{"predicted_action": "respond" | "silent", "answer_text": "...",
 "tool_calls": [{"name": "...", "arguments": {...}}]}
```

## License

MIT — see [LICENSE](LICENSE).
