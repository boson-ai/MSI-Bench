<h1 align="center">MSI-Bench</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2609.24812"><img src="https://img.shields.io/badge/arXiv-2609.24812-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/datasets/M2cha4l1124/MSI-Bench"><img src="https://img.shields.io/badge/HuggingFace-Datasets-yellow.svg" alt="HuggingFace Datasets"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://boson-ai.github.io/MSI-Bench/"><img src="https://img.shields.io/badge/Demo-GitHub%20Pages-2ea44f.svg" alt="Demo"></a>
</p>

## Updates

- **2026-11-12** MSI-Bench v2 released: 624 cases, 2 to 4 speakers, a full-duplex floor layer and real room acoustics. Published under `v2/` on the [HuggingFace dataset](https://huggingface.co/datasets/M2cha4l1124/MSI-Bench/tree/main/v2) (CC BY-NC 4.0). v1 stays at the dataset root.
- **2026-09-21** MSI-Bench v1 released with the paper ([arXiv:2609.24812](https://arxiv.org/abs/2609.24812)): 1,152 cases, 2 or 3 speakers (CC BY 4.0).

> [!WARNING]
> **v1 and v2 scores are not comparable.** v2 has different cases, longer multi-speaker room audio, a different answer protocol and a different judge. Report them separately and always name the version.

Evaluation code for **MSI-Bench** (Multi-Speaker Interaction Benchmark): the
prediction runners, deterministic scoring, LLM answer judge, and the table
aggregation used in the paper *MSI-Bench: Evaluating Multi-Speaker Voice
Interaction for Collaborative AI Agents*.

MSI-Bench tests whether a voice assistant in a multi-speaker conversation can
track who said what, to whom it applies, and who may change it, and then take
the correct next action: answer, call a tool, refuse, ask, or stay silent. It
has 1,152 base test cases over 8 behavior settings
and 6 patterns, with 2 or 3 human speakers per case. **MSI-Bench v2** (624
cases, 2 to 4 speakers, full-duplex floor layer, real room acoustics) is
published; see
[What's new in v2](#whats-new-in-v2).

## Leaderboard

**v1 and v2 scores are not comparable**: different cases, audio, protocol and judge. Each chart stands on its own.

<p align="center"><img src="docs/img/leaderboard_v2.png" alt="MSI-Bench v2 leaderboard: all-pass rate by language" width="820"></p>

v2: 624 cases, English and Mandarin APR per model, judge gpt-6-luna. Full
columns (ARS, Tool, per-pattern APR) are in the v2 results table below.

<p align="center"><img src="docs/img/leaderboard_v1.png" alt="MSI-Bench v1 leaderboard: all-pass rate by language" width="820"></p>

v1: 1,152 cases, results from the paper. † marks thinking or reasoning enabled.

<details>
<summary>v1 results tables (English, Mandarin)</summary>

English results from the paper. APR has a 95% Wilson interval. Bold marks the
best hosted and the best open-weight result per column. † marks reasoning or
thinking enabled. ‡ marks configurations affected by output-format errors.
"–" means the probe was not run or the model produced no valid prediction.

| Model | APR | ARS | Tool | BIR ↑ | PRR ↓ | Auth. | Discl. | Prior. | Seq. | Scope | Retr. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| *Hosted* | | | | | | | | | | | |
| Gemini 3.1 Pro † | **66.8 ±3.8** | **78.0** | **58.8** | **86.5** | **21.5** | 76.0 | **85.4** | 69.8 | **77.1** | 72.9 | **19.8** |
| Gemini 3.5 Flash † | 63.2 ±3.9 | 73.6 | 55.5 | 74.9 | 48.2 | 79.2 | 67.7 | **77.1** | 71.9 | **74.0** | 9.4 |
| GPT Realtime 2.1 (xhigh) † | 58.3 ±4.0 | 73.7 | 44.8 | 83.6 | 28.1 | **86.5** | 82.3 | 70.8 | 51.0 | 50.0 | 9.4 |
| GPT Realtime 2.1 (medium) † | 55.4 ±4.0 | 70.1 | 41.8 | 81.5 | 35.6 | 81.3 | 83.3 | 68.8 | 41.7 | 43.8 | 13.5 |
| GPT Audio 1.5 | 49.5 ±4.1 | 64.4 | 40.3 | 70.2 | 52.4 | 78.1 | 63.5 | 55.2 | 47.9 | 37.5 | 14.6 |
| GPT Audio | 43.9 ±4.0 | 61.5 | 41.8 | 64.1 | 62.8 | 66.7 | 54.2 | 50.0 | 44.8 | 33.3 | 14.6 |
| GPT Audio Mini | 20.7 ±3.3 | 40.0 | 26.3 | – | 97.9 | 30.2 | 11.5 | 38.5 | 20.8 | 15.6 | 7.3 |
| *Open-weight* | | | | | | | | | | | |
| Qwen3-Omni-30B | **34.0 ±3.9** | **53.3** | **26.5** | 43.6 | 92.2 | **63.5** | **57.3** | **33.3** | **27.1** | **15.6** | **7.3** |
| Gemma 4-12B | 22.6 ±3.4 | 43.9 | 20.0 | 57.5 | 85.9 | 42.7 | 38.5 | 26.0 | 12.5 | 13.5 | 2.1 |
| Gemma 4-12B † | 19.8 ±3.2 | 41.3 | 18.3 | **60.9** | **66.1** | 37.5 | 33.3 | 24.0 | 10.4 | 11.5 | 2.1 |
| Qwen2.5-Omni-7B | 16.1 ±3.0 | 38.2 | 10.0 | – | 100.0 | 30.2 | 43.8 | 15.6 | 2.1 | 3.1 | 2.1 |
| MiMo-Audio-7B † ‡ | 14.6 ±2.9 | 33.4 | 6.0 | – | 86.6 | 39.6 | 33.3 | 11.5 | 2.1 | 1.0 | 0.0 |
| MiMo-Audio-7B ‡ | 7.8 ±2.2 | 24.6 | 8.3 | – | 98.8 | 21.9 | 5.2 | 11.5 | 4.2 | 2.1 | 2.1 |
| Phi-4-Multimodal | 5.4 ±1.9 | 32.0 | 0.0 | – | 100.0 | 0.0 | 32.3 | 0.0 | 0.0 | 0.0 | 0.0 |
| Qwen2-Audio-7B | 1.6 ±1.1 | 11.8 | 0.0 | – | 5.7 | 0.0 | 9.4 | 0.0 | 0.0 | 0.0 | 0.0 |

**Mandarin**

| Model | APR | ARS | Tool | BIR ↑ | PRR ↓ | Auth. | Discl. | Prior. | Seq. | Scope | Retr. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| *Hosted* | | | | | | | | | | | |
| Gemini 3.1 Pro † | 52.8 | 69.9 | **58.8** | **74.6** | **32.5** | 80.2 | 53.1 | 51.0 | **71.9** | 59.4 | 1.0 |
| Gemini 3.5 Flash † | 50.2 | 67.5 | **58.8** | 67.4 | 74.9 | 81.3 | 35.4 | **58.3** | 64.6 | **61.5** | 0.0 |
| GPT Realtime 2.1 (xhigh) † | **54.5** | **76.4** | 54.1 | – | 37.8 | **87.5** | **71.9** | **58.3** | 51.0 | 56.3 | 2.1 |
| GPT Realtime 2.1 (medium) † | 49.3 | 70.2 | 47.6 | – | 43.1 | 85.4 | 67.7 | 51.0 | 44.8 | 44.8 | 2.1 |
| GPT Audio 1.5 | 42.5 | 59.6 | 51.0 | 53.3 | 70.2 | 67.7 | 39.6 | 46.9 | 51.0 | 45.8 | **4.2** |
| GPT Audio | 41.1 | 60.1 | 46.5 | 60.9 | 80.6 | 74.0 | 39.6 | 49.0 | 44.8 | 35.4 | **4.2** |
| GPT Audio Mini | 16.5 | 36.6 | 22.7 | – | 97.9 | 27.1 | 20.8 | 27.1 | 7.3 | 15.6 | 1.0 |
| *Open-weight* | | | | | | | | | | | |
| Qwen3-Omni-30B | **19.3** | **41.8** | **26.3** | 43.3 | 99.5 | **28.1** | 19.8 | **35.4** | **17.7** | **11.5** | **3.1** |
| Gemma 4-12B | 4.5 | 28.3 | 0.0 | **60.0** | 92.6 | 0.0 | 27.1 | 0.0 | 0.0 | 0.0 | 0.0 |
| Gemma 4-12B † | 2.3 | 16.7 | 0.0 | 45.5 | **39.3** | 0.0 | 13.5 | 0.0 | 0.0 | 0.0 | 0.0 |
| Qwen2.5-Omni-7B | 8.9 | 33.8 | 10.9 | – | 100.0 | 16.7 | 16.7 | 13.5 | 0.0 | 5.2 | 1.0 |
| MiMo-Audio-7B † ‡ | 7.3 | 23.5 | 3.4 | – | – | 17.7 | 18.8 | 5.2 | 1.0 | 0.0 | 1.0 |
| MiMo-Audio-7B ‡ | 8.9 | 35.8 | 3.9 | – | – | 22.9 | 22.9 | 5.2 | 0.0 | 1.0 | 1.0 |
| Phi-4-Multimodal | 6.4 | 33.7 | 0.3 | – | 99.5 | 1.0 | **36.5** | 1.0 | 0.0 | 0.0 | 0.0 |
| Qwen2-Audio-7B | 0.5 | 12.4 | 0.0 | – | 0.0 | 0.0 | 3.1 | 0.0 | 0.0 | 0.0 | 0.0 |


</details>

<details>
<summary>v2 results table (624 cases, judge gpt-6-luna)</summary>

| Model | APR (±95%) | ARS | Tool | Format errors | Auth. | Discl. | Prior. | Seq. | Scope | Retr. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gemini 3.8 Flash † | 41.8 ±3.9 | 61.0 | 45.3 | 0 | 51.0 | 80.8 | 52.9 | 26.0 | 20.2 | 20.2 |
| GPT Realtime 2.1 (medium) † | 35.3 ±3.7 | 57.0 | 38.0 | 0 | 50.0 | 68.3 | 52.9 | 9.6 | 20.2 | 10.6 |
| Qwen3-Omni-30B | 21.2 ±3.2 | 40.3 | 22.4 | 39 | 33.7 | 56.7 | 23.1 | 2.9 | 0.0 | 10.6 |
| GPT Audio Mini | 18.3 ±3.0 | 38.2 | 21.3 | 0 | 33.7 | 47.1 | 23.1 | 1.0 | 1.0 | 3.8 |
| MiMo-Audio-7B | 12.2 ±2.6 | 34.6 | 11.6 | 15 | 22.1 | 40.4 | 8.7 | 0.0 | 0.0 | 1.9 |
| Qwen2.5-Omni-7B | 11.9 ±2.5 | 32.0 | 12.4 | 39 | 24.0 | 35.6 | 5.8 | 1.9 | 0.0 | 3.8 |
| MiMo-Audio-7B † | 11.4 ±2.5 | 34.0 | 11.3 | 48 | 20.2 | 33.7 | 13.5 | 0.0 | 0.0 | 1.0 |
| Phi-4-Multimodal | 5.9 ±1.9 | 34.5 | 1.1 | 12 | 9.6 | 25.0 | 1.0 | 0.0 | 0.0 | 0.0 |
| Gemma 4-12B | 3.8 ±1.5 | 23.1 | 1.1 | 2 | 4.8 | 17.3 | 1.0 | 0.0 | 0.0 | 0.0 |
| Gemma 4-12B † | 3.8 ±1.5 | 21.9 | 1.6 | 0 | 2.9 | 18.3 | 1.0 | 0.0 | 0.0 | 1.0 |
| Qwen2-Audio-7B | 0.2 ±0.4 | 14.4 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 |

Protocol: Gemini and GPT Audio through native tool calling (Gemini with its
default thinking), GPT Realtime through the Realtime API with reasoning effort
medium, open-weight models through vLLM with schema-constrained decoding
(`IB_LOCAL_GUIDED_JSON=1`); one audio message per room block, earlier assistant
replies as text; judge gpt-6-luna. † thinking or reasoning enabled. Format
errors are cases with no parseable prediction (counted as failures).

</details>

**Metrics** (all in %):

- **APR** (All-Pass Rate): cases that pass every atomic rubric.
- **ARS** (Atomic Rubric Score): atomic rubrics passed, pooled over cases.
- **Tool**: strict pass rate on cases that need a tool call (names, count,
  arguments, ownership, no extra calls).
- **BIR** (Bystander Interference Robustness): answers that still pass after
  an unrelated human-to-human utterance plays while the assistant answers.
- **PRR** (Premature Response Rate): truncated cases where the assistant
  speaks before anyone addresses it. Lower is better. Qwen2-Audio-7B's low PRR
  comes from near-total silence, not from good turn-taking.
- **Patterns**: Auth. = speaker authority constraint, Discl. = selective
  disclosure, Prior. = constraint prioritization, Seq. = sequential constraint
  integration, Scope = scope tracking, Retr. = background speech retrieval.

## What's new in v2

MSI-Bench v2 keeps the six interaction patterns of v1 and rebuilds the audio
around what a room-scale assistant actually hears. 624 cases, two languages.

**A full-duplex floor layer.** Backchannels ("mm-hm", "right"),
interruptions, cut-off sentences, failed attempts to take the floor, and the
assistant's own backchannels in the heard mix. Earlier assistant replies are
part of the room audio history, in the assistant's own voice. About a third of
the cases add an unrelated background conversation between people who are not
part of the task; those voices take their own speaker labels, so the model has
to tell who is talking to it.

**Real room acoustics .** Instead of DSP reverb, we use room impulse responses (RIRs) to render the audio, so that each sound carries the reflections, reverberation and coloration of a real physical space.

**Five times the voices, three times the ambiences.** 480 cloned Common Voice
voices and 249 Freesound ambience recordings. Every ambience
excerpt had its foreground speech removed with a source-separation model
(BS-RoFormer vocal stem subtraction) before mixing, so the only voices in a case
are the ones the case puts there.

**Fewer cases, more coverage, cheaper to run.** 624 cases instead of 1,152. At
equal sample size v2 uses 4 to 5 times as many voices, 2.7 times as many
ambiences, and its individual utterances are more semantically diverse (Vendi
score), so a run represents more conditions for roughly half the audio.

**Tools the way models see them today.** Tool definitions ship in Anthropic
`input_schema` form together with the system prompt used at generation time.
In the scope-tracking and sequential-constraint patterns, gold tool calls bind
each constraint to its owner by voice label (`<|S1|>`, `<|S2|>`, …, numbered in
order of first appearance), so speaker attribution is scored inside the tool
call itself.

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
