# Repo-local serving for audio models

This directory keeps MSI-Bench serving instructions self-contained. It records the known-good vLLM settings for the local audio models under test.

## Supported models

| Key | Default port | Default GPUs | Default TP | API model name |
| --- | ---: | --- | ---: | --- |
| `qwen3-omni-thinking` | 8013 | `0,1` | 2 | `Qwen3-Omni-30B-A3B-Instruct` |
| `qwen3-omni-instruct` | 8013 | `0,1` | 2 | `Qwen3-Omni-30B-A3B-Instruct` |
| `gemma4-12b` | 8014 | `0` | 1 | `gemma-4-12B-it` |
| `mimo-audio` | 8015 | `0` | 1 | `MiMo-Audio-7B-Instruct` |
| `voxtral-small-24b` | 8016 | `0,1` | 2 | `Voxtral-Small-24B-2507` |
| `kimi-audio` | 8017 | `0` | 1 | `Kimi-Audio-7B-Instruct` |
| `qwen2.5-omni` | 8018 | `0` | 1 | `Qwen2.5-Omni-7B` |
| `phi4-multimodal` | 8019 | `0` | 1 | `Phi-4-multimodal-instruct` |
| `qwen2-audio` | 8020 | `0` | 1 | `Qwen2-Audio-7B-Instruct` |

The Qwen Thinking checkpoint is served with the same `served-model-name` as the Qwen config (`Qwen3-Omni-30B-A3B-Instruct`). Use that model id in OpenAI-compatible clients unless you edit the config.

Gemma 4 and MiMo use one checkpoint for both leaderboard variants. The evaluator
passes `chat_template_kwargs.enable_thinking=true` for the dagger target and
`false` for the plain target. Voxtral receives the benchmark system instructions
inside its first user message because audio requests do not support a system role.
The Gemma 4 serving backend accepts multiple audio clips per request
(`limit-mm-per-prompt` audio: 12, matching the suite's max clip count); the
`vllm-gemma4-12b-20260604` image is required because older images ship a
Transformers that does not recognize the `gemma4_unified` checkpoint architecture.

## Run the registered local eval targets

Start only the model service you intend to evaluate, then select that target from
the opt-in local declaration file. For example:

```bash
serve/serve_model.sh qwen2-audio --gpus 6
serve/wait_ready.sh 8020 900

uv run ib eval --run <validation-run> \
  --target turn_based:local:Qwen2-Audio-7B-Instruct \
  --target-workers 1
```

The model loader skips declarations whose `checkpoint_path` is absent. The file
is separate from `configs/eval-models.yaml` so the default cloud evaluation does
not require every local server to be online. If all local declarations are used,
start their endpoints on the default ports above; otherwise copy the desired
target into a smaller targets file.

Kimi is present in the inventory and has a serving recipe, but the current vLLM
integration is reliable only through `/v1/audio/transcriptions`. MSI-Bench
fails closed for Kimi rather than combining its ASR with a second model and
mislabeling that cascade as a Kimi-only score.

## Start Qwen3-Omni Thinking

```bash
serve/serve_model.sh qwen3-omni-thinking \
  --gpus 1,2 \
  --tp 2 \
  --port 8013 \
  --name qwen3-omni-thinking
```

Equivalent dry run, useful for review:

```bash
serve/serve_model.sh qwen3-omni-thinking --gpus 1,2 --tp 2 --port 8013 --dry-run
```

Wait for readiness:

```bash
serve/wait_ready.sh 8013 900
```

Use in `golden_test_hard`:

```bash
UV_LINK_MODE=copy uv run --frozen --directory src/experiment python -m golden_test_hard.run \
  --source ../../goals/Gold.draft.txt \
  --out golden_test_hard/output/<run-name> \
  --resume \
  --grade-model gpt5.4 \
  --qwen-base-url http://localhost:8013/v1 \
  --qwen-model Qwen3-Omni-30B-A3B-Instruct
```

Stop:

```bash
serve/stop_model.sh qwen3-omni-thinking
```

## Start Kimi-Audio

```bash
serve/serve_model.sh kimi-audio \
  --gpus 5 \
  --port 8017 \
  --name kimi-audio

serve/wait_ready.sh 8017 900
```

Use in `golden_test_hard` with the ASR+text adapter:

```bash
UV_LINK_MODE=copy uv run --frozen --directory src/experiment python -m golden_test_hard.run \
  --source ../../goals/Gold.draft.txt \
  --out golden_test_hard/output/<run-name> \
  --resume \
  --grade-model gpt5.4 \
  --kimi-base-url http://localhost:8017/v1 \
  --kimi-model Kimi-Audio-7B-Instruct \
  --kimi-mode asr_text
```

Stop:

```bash
serve/stop_model.sh kimi-audio
```

## Why the previous Qwen serve attempts failed

The successful path is not just "run any vLLM". Qwen3-Omni needs an image with Qwen3-Omni support, the right container model path, the right GPU topology, and a real readiness check.

Common failure modes:

1. **Wrong engine/image** — generic or latest vLLM images may not register `Qwen3OmniMoeForConditionalGeneration`. This repo uses the known-good image `registry.canada.boson.ai/eval-runner:vllm-audio-20260403` for Qwen.
2. **Container running is not readiness** — the port can exist while vLLM is still loading weights; `/v1/models` may reset until the server is ready. Always wait with `serve/wait_ready.sh`.
3. **Wrong GPU shape** — Qwen3-Omni 30B should use two A100-40GB GPUs with `--tp 2`. Passing two GPU ids without TP can accidentally create the wrong serving shape in generated commands.
4. **Wrong path namespace** — inside Docker, `/ceph/models` is mounted as `/models`, so use `/models/Qwen3-Omni-30B-A3B-Thinking`, not only the host path.
5. **Kimi API mismatch** — Kimi-Audio is most reliable here through `/v1/audio/transcriptions`; `golden_test_hard` intentionally uses `--kimi-mode asr_text`.

## Local files

- `serve/configs/qwen3-omni/vllm-audio.yaml` — repo-local Qwen vLLM config.
- `serve/configs/kimi-audio/vllm-nightly-audio.yaml` — repo-local Kimi vLLM config.
- `serve/configs/kimi-audio/chat_template.jinja` — Kimi chat template required by the config.
- `serve/configs/<model>/` — repo-local configs for Gemma, MiMo, Voxtral, Qwen2/2.5, and Phi-4.
- `serve/serve_model.sh` — self-contained Docker launcher.
- `serve/wait_ready.sh` — `/v1/models` readiness gate.
- `serve/stop_model.sh` — cleanup helper.
