"""Read numeric columns out of an arbitrary CSV.

No header name is hard-coded anywhere in this module: the caller names every column it
wants.  This function does IO -- never call it from a real-time loop.
"""

from __future__ import annotations

import csv
import logging
import math
from typing import Literal

from ._errors import Sim2RealActuatorError

__all__ = ["load_columns"]

_log = logging.getLogger(__name__)

_DEG2RAD = math.pi / 180.0
VelUnit = Literal["rad/s", "deg/s"]

#: Output key -> the keyword argument the caller named it with, for error messages.
_ARGUMENT_OF = {"t_s": "t", "tau_nm": "tau", "vel_rad_s": "vel", "pos_rad": "pos"}


def _parse(value: str, *, column: str, row: int) -> float:
    text = value.strip() if isinstance(value, str) else value
    if text is None or text == "":
        raise Sim2RealActuatorError(
            f"empty cell in column {column!r} at data row {row} (1-based, header excluded)"
        )
    try:
        return float(text)
    except (TypeError, ValueError) as exc:
        raise Sim2RealActuatorError(
            f"column {column!r} at data row {row} is not a number: {value!r}"
        ) from exc


def load_columns(
    path: str,
    *,
    t: str,
    tau: str,
    vel: str,
    pos: str | None = None,
    vel_unit: VelUnit = "rad/s",
) -> dict[str, list[float]]:
    """Pull the named columns out of ``path`` and return them ready for :func:`identify`.

    Args:
        path: CSV file path.  Must have a header row.
        t: Header name of the timestamp column, seconds.
        tau: Header name of the torque column, Nm.
        vel: Header name of the velocity column, unit given by ``vel_unit``.
        pos: Optional header name of the position column.
        vel_unit: ``"rad/s"`` (default) or ``"deg/s"``.  ``"deg/s"`` converts the
            velocity column **and** the position column to radians -- on real hardware
            both come off the same encoder in the same unit.

    Returns:
        ``{"t_s": [...], "tau_nm": [...], "vel_rad_s": [...]}`` plus ``"pos_rad"`` when
        ``pos`` is given.  The keys are exactly :func:`identify`'s keyword arguments, so
        ``identify(**load_columns(...))`` works.

    Raises:
        ValueError: Missing header row, a requested column that is not in the header,
            an empty or non-numeric cell, an unknown ``vel_unit``, or an empty file.
        FileNotFoundError: If ``path`` does not exist.
    """
    if vel_unit not in ("rad/s", "deg/s"):
        raise Sim2RealActuatorError(f"vel_unit must be 'rad/s' or 'deg/s', got {vel_unit!r}")
    for name, value in (("t", t), ("tau", tau), ("vel", vel)):
        if not isinstance(value, str) or not value:
            raise Sim2RealActuatorError(f"{name}= must be a non-empty column name, got {value!r}")
    if pos is not None and (not isinstance(pos, str) or not pos):
        raise Sim2RealActuatorError(f"pos= must be a non-empty column name or None, got {pos!r}")

    wanted: dict[str, str] = {"t_s": t, "tau_nm": tau, "vel_rad_s": vel}
    if pos is not None:
        wanted["pos_rad"] = pos

    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames
        if not header:
            raise Sim2RealActuatorError(f"{path!r} has no header row")
        missing = {_ARGUMENT_OF[key]: name for key, name in wanted.items() if name not in header}
        if missing:
            raise Sim2RealActuatorError(
                f"{path!r} is missing column(s) "
                + ", ".join(f"{arg}={name!r}" for arg, name in sorted(missing.items()))
                + f". Available columns: {list(header)}"
            )
        out: dict[str, list[float]] = {key: [] for key in wanted}
        row_index = 0
        for row in reader:
            row_index += 1
            for key, name in wanted.items():
                out[key].append(_parse(row.get(name), column=name, row=row_index))

    if row_index == 0:
        raise Sim2RealActuatorError(f"{path!r} has a header but no data rows")

    if vel_unit == "deg/s":
        out["vel_rad_s"] = [v * _DEG2RAD for v in out["vel_rad_s"]]
        if "pos_rad" in out:
            out["pos_rad"] = [v * _DEG2RAD for v in out["pos_rad"]]

    _log.debug("loaded %d rows x %d columns from %s", row_index, len(out), path)
    return out
