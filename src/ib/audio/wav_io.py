"""wav_io — stdlib PCM16 WAV read/write helpers (mono, plus a multi-channel writer).

Calling spec:
    write_mono_pcm16_wav(path, samples, sample_rate)
    sample_rate, samples = read_mono_pcm16_wav(path)
    write_pcm16_channels_wav(path, channels, sample_rate)   one channel per equal-length sequence

Samples are floats in [-1.0, 1.0]. Persisted bytes are canonical mono PCM16 WAV;
unsupported layouts fail loudly instead of being silently coerced.
"""

from __future__ import annotations

import wave
from pathlib import Path
from collections.abc import Sequence
from typing import Iterable

import numpy as np

from ib.audio.sample_array import as_samples, clip_samples


class WavFormatError(ValueError):
    """Raised when a WAV file is not canonical mono PCM16."""


def write_mono_pcm16_wav(path: str | Path, samples: Iterable[float], sample_rate: int) -> None:
    """Write canonical mono PCM16 WAV bytes: each sample clipped to [-1, 1], scaled by 32767
    and rounded half to even; -1.0 is written as -32768."""
    if sample_rate <= 0:
        raise WavFormatError("sample_rate must be > 0")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    clipped = clip_samples(as_samples(samples))
    scaled = np.rint(clipped * 32767.0)
    scaled[clipped <= -1.0] = -32768.0
    pcm = scaled.astype("<i2")
    with wave.open(str(destination), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def write_pcm16_channels_wav(path: str | Path, channels: Sequence[Iterable[float]], sample_rate: int) -> None:
    """Write an interleaved multi-channel PCM16 WAV, one channel per sequence (all the same length).

    Each channel is quantized exactly as write_mono_pcm16_wav quantizes a mono clip.
    """
    if sample_rate <= 0:
        raise WavFormatError("sample_rate must be > 0")
    if not channels:
        raise WavFormatError("a multi-channel WAV needs at least one channel")
    arrays = [_pcm16(channel) for channel in channels]
    if len({len(array) for array in arrays}) != 1:
        raise WavFormatError("every channel of a multi-channel WAV needs the same length")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(destination), "wb") as handle:
        handle.setnchannels(len(arrays))
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(np.stack(arrays, axis=1).tobytes())


def _pcm16(samples: Iterable[float]) -> np.ndarray:
    clipped = clip_samples(as_samples(samples))
    scaled = np.rint(clipped * 32767.0)
    scaled[clipped <= -1.0] = -32768.0
    return scaled.astype("<i2")


def read_mono_pcm16_wav(path: str | Path) -> tuple[int, list[float]]:
    """Read canonical mono PCM16 WAV bytes and return sample_rate plus float samples
    (each sample / 32767; -32768 reads as -1.0)."""
    source = Path(path)
    with wave.open(str(source), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        compression = handle.getcomptype()
        if channels != 1:
            raise WavFormatError(f"expected mono WAV (1 channel), got {channels}")
        if sample_width != 2:
            raise WavFormatError(f"expected PCM16 WAV (2-byte samples), got {sample_width}")
        if compression != "NONE":
            raise WavFormatError(f"expected uncompressed PCM WAV, got {compression}")
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    pcm = np.frombuffer(frames, dtype="<i2")
    samples = pcm / 32767.0
    samples[pcm == -32768] = -1.0
    return sample_rate, samples.tolist()
