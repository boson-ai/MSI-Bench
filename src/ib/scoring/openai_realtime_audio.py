"""openai_realtime_audio — deterministic PCM conversion for Realtime APIs.

Calling spec:
    wav_as_pcm16_base64(path) -> 24 kHz mono PCM16 as base64 text
    float_samples_to_pcm16(samples) -> little-endian PCM16 bytes
    resample(samples, source_rate, target_rate) -> resampled float samples
    pcm16_delta_ms(delta) -> duration of a base64 PCM16 delta in milliseconds

All functions are deterministic and have no side effects beyond reading the
explicit WAV input path.
"""

from __future__ import annotations

import base64
from pathlib import Path

from ib.audio.wav_io import read_mono_pcm16_wav

DEFAULT_REALTIME_SAMPLE_RATE = 24_000


def wav_as_pcm16_base64(path: Path) -> str:
    sample_rate, samples = read_mono_pcm16_wav(path)
    if sample_rate != DEFAULT_REALTIME_SAMPLE_RATE:
        samples = resample(samples, sample_rate, DEFAULT_REALTIME_SAMPLE_RATE)
    return base64.b64encode(float_samples_to_pcm16(samples)).decode("ascii")


def float_samples_to_pcm16(samples: list[float]) -> bytes:
    out = bytearray()
    for sample in samples:
        clipped = max(-1.0, min(1.0, float(sample)))
        value = int(round(clipped * 32767.0))
        if value < 0:
            value += 65536
        out.extend(value.to_bytes(2, "little", signed=False))
    return bytes(out)


def resample(samples: list[float], source_rate: int, target_rate: int) -> list[float]:
    if source_rate <= 0 or source_rate == target_rate or not samples:
        return list(samples)
    target_len = max(1, round(len(samples) * target_rate / source_rate))
    if target_len == 1:
        return [samples[0]]
    scale = (len(samples) - 1) / (target_len - 1)
    return [samples[round(index * scale)] for index in range(target_len)]


def pcm16_delta_ms(delta: str) -> int:
    if not delta:
        return 0
    try:
        data = base64.b64decode(delta)
    except ValueError:
        return 0
    return round((len(data) / 2) * 1000 / DEFAULT_REALTIME_SAMPLE_RATE)
