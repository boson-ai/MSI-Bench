"""wav_io — stdlib mono PCM16 WAV read/write helpers.

Calling spec:
    write_mono_pcm16_wav(path, samples, sample_rate)
    sample_rate, samples = read_mono_pcm16_wav(path)

Samples are floats in [-1.0, 1.0]. Persisted bytes are canonical mono PCM16 WAV;
unsupported layouts fail loudly instead of being silently coerced.
"""

from __future__ import annotations

import wave
from array import array
from pathlib import Path
from typing import Iterable


class WavFormatError(ValueError):
    """Raised when a WAV file is not canonical mono PCM16."""


def _clip(sample: float) -> float:
    return max(-1.0, min(1.0, float(sample)))


def _float_to_i16(sample: float) -> int:
    value = _clip(sample)
    if value >= 1.0:
        return 32767
    if value <= -1.0:
        return -32768
    return int(round(value * 32767.0))


def _i16_to_float(sample: int) -> float:
    if sample == -32768:
        return -1.0
    return sample / 32767.0


def write_mono_pcm16_wav(path: str | Path, samples: Iterable[float], sample_rate: int) -> None:
    """Write canonical mono PCM16 WAV bytes using only stdlib wave."""
    if sample_rate <= 0:
        raise WavFormatError("sample_rate must be > 0")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pcm = array("h", (_float_to_i16(sample) for sample in samples))
    if pcm.itemsize != 2:
        raise WavFormatError("platform array('h') is not 16-bit")
    if pcm.itemsize == 2 and _is_big_endian_array():
        pcm.byteswap()
    with wave.open(str(destination), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def read_mono_pcm16_wav(path: str | Path) -> tuple[int, list[float]]:
    """Read canonical mono PCM16 WAV bytes and return sample_rate plus float samples."""
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
    pcm = array("h")
    pcm.frombytes(frames)
    if _is_big_endian_array():
        pcm.byteswap()
    return sample_rate, [_i16_to_float(sample) for sample in pcm]


def _is_big_endian_array() -> bool:
    marker = array("h", [1]).tobytes()
    return marker == b"\x00\x01"
