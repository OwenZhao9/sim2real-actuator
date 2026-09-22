#!/usr/bin/env python3
"""Identify a real hip-exoskeleton's actuators from the recordings in ``examples/data/``.

Run it with::

    uv run python examples/identify_exoskeleton.py

Everything project-specific lives *here*, in the caller: the CSV header names, which
column belongs to which joint, and how a break-away ramp is sliced out of a log.  The
library itself knows none of it -- ``load_columns`` takes column names as arguments.
"""

from __future__ import annotations

import json
from pathlib import Path

from sim2real_actuator import ActuatorParams, breakaway, identify, load_columns

DATA = Path(__file__).parent / "data"

# Column layout of this particular recorder.  Nothing in the library knows these names.
COLUMNS = {
    "left": {"tau": "cmd_l", "vel": "ldps", "pos": "ldeg"},
    "right": {"tau": "cmd_r", "vel": "rdps", "pos": "rdeg"},
}
TIME_COLUMN = "host_t"
VEL_UNIT = "deg/s"

# The velocity column is quantised to 0.1 deg/s and sampled at ~5 ms, so differentiating
# it is mostly noise and the inertia term is not identifiable from these logs.  Ask for
# the friction model only, and state the inertia as an explicit assumption elsewhere.
FIT_INERTIA = False

#: Fraction of the near-peak speed below which a sample counts as "not moving yet".
MOVING_FRAC = 0.02


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q / 100.0 * (len(ordered) - 1))))
    return ordered[idx]


def onset_torques(tau_nm: list[float], vel_rad_s: list[float]) -> tuple[float, float]:
    """Largest torque magnitude still held statically before motion starts, per direction.

    This is a caller-side heuristic over one particular kind of log (a slow torque ramp).
    It deliberately lives in the example, not in the library.
    """
    peak = _percentile([abs(v) for v in vel_rad_s], 99.5)
    threshold = MOVING_FRAC * peak
    moving = [abs(v) > threshold for v in vel_rad_s]
    up: list[float] = []
    down: list[float] = []
    for i in range(3, len(vel_rad_s)):
        started = moving[i] and moving[i - 1] and not moving[i - 2] and not moving[i - 3]
        if started:
            (up if vel_rad_s[i] > 0.0 else down).append(abs(tau_nm[i - 2]))
    return (max(up) if up else 0.0, max(down) if down else 0.0)


def main() -> int:
    recordings = sorted(DATA.glob("*.csv"))
    if not recordings:
        print(f"no CSV files in {DATA}")
        return 1

    print("=" * 96)
    print("Least-squares identification   tau ~ viscous*w + coulomb*sign(w) - bias")
    print("=" * 96)
    print(
        f"{'recording':32s} {'joint':6s} {'coulomb_nm':>11s} {'viscous':>10s} "
        f"{'bias_nm':>9s} {'samples':>9s}"
    )
    print("-" * 96)

    identified: dict[str, ActuatorParams] = {}
    for csv_path in recordings:
        for joint, cols in COLUMNS.items():
            data = load_columns(
                str(csv_path),
                t=TIME_COLUMN,
                tau=cols["tau"],
                vel=cols["vel"],
                pos=cols["pos"],
                vel_unit=VEL_UNIT,
            )
            if max(abs(x) for x in data["tau_nm"]) == 0.0:
                print(f"{csv_path.stem:32s} {joint:6s}  (no torque commanded -- skipped)")
                continue
            try:
                params = identify(**data, fit_inertia=FIT_INERTIA)
            except ValueError as exc:
                print(f"{csv_path.stem:32s} {joint:6s}  skipped: {exc}")
                continue
            used = params.source.split("n=")[1].split(",")[0]
            print(
                f"{csv_path.stem:32s} {joint:6s} {params.coulomb_nm:11.4f} "
                f"{params.viscous_nm_s_per_rad:10.4f} {params.bias_nm:9.4f} {used:>9s}"
            )
            if csv_path.stem == "breakaway-measurement":
                identified[joint] = params

    print()
    print("=" * 96)
    print("Break-away split of the same ramp   breakaway(up, down) -> (static friction, bias)")
    print("=" * 96)
    print(f"{'joint':6s} {'up_nm':>8s} {'down_nm':>9s} {'static_nm':>11s} {'bias_nm':>9s}")
    print("-" * 96)
    ramp = DATA / "breakaway-measurement.csv"
    for joint, cols in COLUMNS.items():
        data = load_columns(
            str(ramp),
            t=TIME_COLUMN,
            tau=cols["tau"],
            vel=cols["vel"],
            pos=cols["pos"],
            vel_unit=VEL_UNIT,
        )
        up, down = onset_torques(data["tau_nm"], data["vel_rad_s"])
        static, bias = breakaway(up, down)
        print(f"{joint:6s} {up:8.3f} {down:9.3f} {static:11.3f} {bias:9.3f}")

    print()
    print("Note: this recorder's left encoder counts the mirror image of the right one, so")
    print("the left joint's bias comes out negative here.  Flip it caller-side to compare")
    print("against numbers quoted in the wearer's anatomical frame -- the library reports")
    print("whatever frame the columns it was handed are in, and mirrors nothing itself.")

    print()
    print("=" * 96)
    print("JSON round-trip (these go straight into a log / dashboard / experience store)")
    print("=" * 96)
    for joint, params in identified.items():
        blob = json.dumps(params.to_dict(), sort_keys=True)
        assert ActuatorParams.from_dict(json.loads(blob)) == params
        print(f"{joint:6s} {blob}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
