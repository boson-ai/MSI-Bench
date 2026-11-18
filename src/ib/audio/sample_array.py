"""sample_array — float64 helpers shared by the numpy render kernels.

Each helper reproduces, element by element, the scalar Python expression it replaces, so
the kernels return bitwise the same samples as the per-sample loops they replaced.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def as_samples(samples: Iterable[float]) -> np.ndarray:
    """Return the samples as a float64 array; a float64 array is returned as is."""
    if not hasattr(samples, "__len__"):
        samples = list(samples)
    return np.asarray(samples, dtype=np.float64)


def clip_samples(samples: np.ndarray) -> np.ndarray:
    """Return max(-1.0, min(1.0, x)) per sample; NaN becomes 1.0 as it does there."""
    return np.clip(np.nan_to_num(samples, nan=1.0), -1.0, 1.0)


def squares(samples: np.ndarray) -> np.ndarray:
    """Return float(x) ** 2 per sample.

    Python's ** calls libm pow, which is not always the correctly rounded x * x;
    float_power calls the same pow, so the squares match bit for bit.
    """
    return np.float_power(samples, 2.0)


def sequential_sum(values: np.ndarray) -> float:
    """Return the builtin sum() of the values, which adds them in the same order and way."""
    return sum(values.tolist())


_SPLITTER = 134217729.0  # 2**27 + 1, Veltkamp's constant for float64
_SCALE = 1e8  # 10**8 has 19 significant bits, so the split products below are exact
# "0000".."9999" as little-endian uint32 words, so four digits are written in one store.
_FOUR_DIGITS = np.frombuffer(b"".join(f"{n:04d}".encode() for n in range(10_000)), dtype="<u4")
# One formatted value: sign (NUL when non-negative, dropped at the end), integer digit,
# point, eight decimals, comma.
_FIXED8 = np.dtype(
    [("sign", "u1"), ("integer", "u1"), ("point", "u1"), ("high", "<u4"), ("low", "<u4"), ("comma", "u1")]
)


def fixed8_payload(values: np.ndarray) -> bytes:
    """Return ",".join(f"{x:.8f}" for x in values).encode() for a float64 array.

    Values in [-1, 1] are rounded with integer arithmetic: x * 1e8 is rounded half to even
    on its exact value, recovered as p + e with Dekker's error-free product, exactly as
    Python's correctly rounded formatting does. Other values use Python's formatting.
    """
    if len(values) == 0:
        return b""
    if not np.all(np.abs(values) <= 1.0):
        return ",".join(f"{x:.8f}" for x in values.tolist()).encode("utf-8")
    product = values * _SCALE
    split = _SPLITTER * values
    high = split - (split - values)
    error = (high * _SCALE - product) + (values - high) * _SCALE
    units = np.rint(product)
    floor = np.floor(product)
    tie = product - floor == 0.5
    units[tie & (error > 0.0)] = floor[tie & (error > 0.0)] + 1.0
    units[tie & (error < 0.0)] = floor[tie & (error < 0.0)]
    magnitude = np.abs(units).astype(np.int32)
    decimals = magnitude % 100_000_000
    text = np.empty(len(values), dtype=_FIXED8)
    text["sign"] = np.where(np.signbit(values), ord("-"), 0)
    text["integer"] = ord("0") + magnitude // 100_000_000
    text["point"] = ord(".")
    text["high"] = _FOUR_DIGITS[decimals // 10_000]
    text["low"] = _FOUR_DIGITS[decimals % 10_000]
    text["comma"] = ord(",")
    return text.tobytes()[:-1].replace(b"\0", b"")
