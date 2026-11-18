# Serving the open-weight models

`serve/configs/<model>/` holds the vLLM settings the published open-weight runs used. Each file is a
`vllm serve --config` file. Replace `<path-to-checkpoint>` with the directory that holds the model's
weights; a Hugging Face model id works too where the file has no LoRA paths. Keep `served-model-name`
unchanged: `ib eval` sends it as the model id and looks it up in `src/ib/scoring/local_audio_models.py`.

| Config | Served model name | `ib eval` reads the base URL from | Default |
| --- | --- | --- | --- |
| `qwen3-omni/vllm-audio.yaml` | `Qwen3-Omni-30B-A3B-Instruct` | `QWEN_UNDERSTANDING_BASE_URL` | `http://localhost:8013/v1` |
| `gemma4-12b/vllm-gemma4.yaml` | `gemma-4-12B-it` | `GEMMA4_UNDERSTANDING_BASE_URL` | `http://localhost:8014/v1` |
| `mimo-audio/vllm-omni.yaml` | `MiMo-Audio-7B-Instruct` | `MIMO_AUDIO_UNDERSTANDING_BASE_URL` | `http://localhost:8015/v1` |
| `voxtral-small/vllm-audio.yaml` | `Voxtral-Small-24B-2507` | `VOXTRAL_UNDERSTANDING_BASE_URL` | `http://localhost:8016/v1` |
| `qwen2.5-omni/vllm-audio.yaml` | `Qwen2.5-Omni-7B` | `QWEN25_OMNI_UNDERSTANDING_BASE_URL` | `http://localhost:8018/v1` |
| `phi4-multimodal/vllm-audio.yaml` | `Phi-4-multimodal-instruct` | `PHI4_MULTIMODAL_UNDERSTANDING_BASE_URL` | `http://localhost:8019/v1` |
| `qwen2-audio/vllm-audio.yaml` | `Qwen2-Audio-7B-Instruct` | `QWEN2_AUDIO_UNDERSTANDING_BASE_URL` | `http://localhost:8020/v1` |

`LOCAL_AUDIO_UNDERSTANDING_BASE_URL` applies to every local model whose own variable is unset. The
files listen on port 8000; pass `--port` to match the default above or set the variable.

```bash
vllm serve --config serve/configs/qwen2-audio/vllm-audio.yaml --port 8020
export IB_LOCAL_GUIDED_JSON=1          # schema-constrained decoding, as in the published runs
ib eval --run <suite> --models configs/eval-targets/qwen2-audio.yaml --output <dir> \
    --answer-judge --judge-provider openrouter --judge-model <judge>
```

Notes:

- Gemma 4 and MiMo-Audio serve both table variants from one checkpoint. The target file sets
  `thinking_mode`, and the evaluator passes `chat_template_kwargs.enable_thinking` per request.
- Voxtral receives the setup prompt inside its first user message, because its audio requests do not
  take a system role.
- MiMo-Audio is served with vLLM-Omni (`vllm-omni.yaml`, chat template in the same directory).
- Kimi-Audio has a serving file but is not evaluated: its vLLM route only transcribes, and the evaluator
  refuses it rather than scoring an ASR cascade as Kimi.
