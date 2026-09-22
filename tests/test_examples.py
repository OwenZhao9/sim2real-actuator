"""The examples are acceptance criteria, so CI runs them for real."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
EXAMPLES = ROOT / "examples"
DATA = EXAMPLES / "data"


def run_example(name: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(EXAMPLES / name)],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
        cwd=ROOT,
    )


def test_the_recorded_csvs_are_present():
    names = sorted(p.name for p in DATA.glob("*.csv"))
    assert names == [
        "breakaway-measurement.csv",
        "direction-calibration-left.csv",
        "direction-calibration-right.csv",
    ]


def test_identify_exoskeleton_runs_on_the_real_recordings():
    result = run_example("identify_exoskeleton.py")
    assert result.returncode == 0, result.stderr
    assert "breakaway-measurement" in result.stdout
    assert "static_nm" in result.stdout


def test_digital_twin_matches_the_real_recording_in_scale():
    """The hard acceptance item: a two-joint digital body driven by identified params."""
    result = run_example("digital_twin.py")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ACCEPTED" in result.stdout
    assert "OFF BY MORE THAN 10x" not in result.stdout


def test_the_digital_twin_offers_the_real_bridges_surface():
    sys.path.insert(0, str(EXAMPLES))
    try:
        import digital_twin  # imported late on purpose
    finally:
        sys.path.pop(0)

    left, right, _ = digital_twin.identified_params()
    twin = digital_twin.DigitalTwin(left, right, start_ldeg=-36.0, start_rdeg=34.0)
    for method in (
        "open",
        "ping",
        "version",
        "on_sample",
        "set_torque",
        "commanded",
        "stream_hz",
        "enable",
        "disable",
        "close",
    ):
        assert callable(getattr(twin, method)), method

    seen: list[object] = []
    twin.open().on_sample(seen.append)
    twin.enable(send_torque=True)
    twin.set_torque(0.6, -0.6)
    assert twin.commanded() == (0.6, -0.6)
    sample = twin.advance(0.005)
    assert seen == [sample]
    assert sample.cmd_l == pytest.approx(0.6)
    twin.close()
    assert twin.commanded() == (0.0, 0.0)


def test_the_digital_twin_is_reproducible():
    sys.path.insert(0, str(EXAMPLES))
    try:
        import digital_twin
    finally:
        sys.path.pop(0)

    left, right, _ = digital_twin.identified_params()

    def run() -> list[tuple[float, float]]:
        twin = digital_twin.DigitalTwin(left, right, start_ldeg=-36.0, start_rdeg=34.0).open()
        twin.enable(send_torque=True)
        out = []
        for i in range(500):
            twin.set_torque(0.5 if i % 100 < 50 else -0.5, 0.3)
            sample = twin.advance(0.005)
            out.append((sample.ldeg, sample.rdps))
        return out

    assert run() == run()


def test_disabled_twin_ignores_torque_commands():
    sys.path.insert(0, str(EXAMPLES))
    try:
        import digital_twin
    finally:
        sys.path.pop(0)

    left, right, _ = digital_twin.identified_params()
    twin = digital_twin.DigitalTwin(left, right, start_ldeg=0.0, start_rdeg=0.0).open()
    twin.set_torque(5.0, -5.0)
    for _ in range(200):
        sample = twin.advance(0.005)
    assert sample.ldeg == pytest.approx(0.0)
    assert sample.cmd_l == 0.0
