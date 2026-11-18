"""S5.A1 — canonical mono PCM16 WAV I/O."""

from __future__ import annotations

import wave

import pytest

from ib.audio.wav_io import WavFormatError, read_mono_pcm16_wav, write_mono_pcm16_wav


def test_mono_pcm16_wav_round_trips_canonical_bytes(tmp_path) -> None:
    path = tmp_path / "sample.wav"
    samples = [-1.0, -0.5, 0.0, 0.5, 1.0]

    write_mono_pcm16_wav(path, samples, 16_000)
    first_bytes = path.read_bytes()
    rate, decoded = read_mono_pcm16_wav(path)
    write_mono_pcm16_wav(tmp_path / "same.wav", samples, 16_000)

    assert rate == 16_000
    assert decoded[0] == -1.0
    assert decoded[-1] == 1.0
    assert decoded[2] == 0.0
    assert (tmp_path / "same.wav").read_bytes() == first_bytes
    with wave.open(str(path), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getcomptype() == "NONE"


def test_samples_are_clipped_to_pcm16_bounds(tmp_path) -> None:
    path = tmp_path / "clipped.wav"

    write_mono_pcm16_wav(path, [-2.0, 2.0], 8_000)
    _rate, decoded = read_mono_pcm16_wav(path)

    assert decoded == [-1.0, 1.0]


def test_rejects_unsupported_stereo_wav(tmp_path) -> None:
    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\x00\x00\x00\x00")

    with pytest.raises(WavFormatError, match="mono"):
        read_mono_pcm16_wav(path)


def test_rejects_non_positive_sample_rate(tmp_path) -> None:
    with pytest.raises(WavFormatError, match="sample_rate"):
        write_mono_pcm16_wav(tmp_path / "bad.wav", [0.0], 0)
