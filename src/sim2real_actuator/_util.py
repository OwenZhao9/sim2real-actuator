"""Internal helpers.  Not part of the public API."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from ._errors import Sim2RealActuatorError

__all__ = [
    "check_finite",
    "check_non_negative",
    "check_positive_or_none",
    "to_float_array",
    "unknown_keys",
]


def check_finite(name: str, value: Any) -> float:
    """Coerce to ``float`` and reject NaN / inf."""
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - message is what matters
        raise Sim2RealActuatorError(f"{name} must be a real number, got {value!r}") from exc
    if not math.isfinite(out):
        raise Sim2RealActuatorError(f"{name} must be finite, got {out!r}")
    return out


def check_non_negative(name: str, value: Any) -> float:
    out = check_finite(name, value)
    if out < 0.0:
        raise Sim2RealActuatorError(f"{name} must be >= 0, got {out!r}")
    return out


def check_positive_or_none(name: str, value: Any) -> float | None:
    if value is None:
        return None
    out = check_finite(name, value)
    if out <= 0.0:
        raise Sim2RealActuatorError(f"{name} must be > 0 when given, got {out!r}")
    return out


def to_float_array(name: str, seq: Sequence[float]) -> np.ndarray:
    """Convert a caller-supplied sequence to a contiguous 1-D float64 array."""
    if seq is None:
        raise Sim2RealActuatorError(f"{name} must be a sequence of numbers, got None")
    try:
        arr = np.asarray(seq, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise Sim2RealActuatorError(f"{name} must be a sequence of real numbers ({exc})") from exc
    if arr.ndim != 1:
        raise Sim2RealActuatorError(f"{name} must be 1-D, got shape {arr.shape}")
    return np.ascontiguousarray(arr)


def unknown_keys(cls_name: str, given: dict[str, Any], allowed: frozenset[str]) -> None:
    extra = sorted(set(given) - allowed)
    if extra:
        raise Sim2RealActuatorError(
            f"{cls_name}.from_dict() got unknown key(s): {extra}. Known keys: {sorted(allowed)}"
        )
