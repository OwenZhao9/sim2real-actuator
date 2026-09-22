"""load_columns(): read any CSV, hard-code no header name."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from sim2real_actuator import identify, load_columns

SAMPLE = "\n".join(
    [
        "stamp,torque,speed,angle,other",
        "0.000,0.10,0.0,0.0,x",
        "0.010,0.20,1.5,0.015,y",
        "0.020,0.30,-2.5,0.010,z",
    ]
)


@pytest.fixture
def csv_path(tmp_path: Path) -> Path:
    path = tmp_path / "recording.csv"
    path.write_text(SAMPLE, encoding="utf-8")
    return path


def test_reads_the_named_columns(csv_path):
    out = load_columns(str(csv_path), t="stamp", tau="torque", vel="speed")
    assert out == {
        "t_s": [0.0, 0.010, 0.020],
        "tau_nm": [0.10, 0.20, 0.30],
        "vel_rad_s": [0.0, 1.5, -2.5],
    }


def test_position_is_optional_and_named_by_the_caller(csv_path):
    out = load_columns(str(csv_path), t="stamp", tau="torque", vel="speed", pos="angle")
    assert out["pos_rad"] == [0.0, 0.015, 0.010]
    assert set(out) == {"t_s", "tau_nm", "vel_rad_s", "pos_rad"}


def test_degrees_per_second_converts_velocity_and_position(csv_path):
    out = load_columns(
        str(csv_path), t="stamp", tau="torque", vel="speed", pos="angle", vel_unit="deg/s"
    )
    assert out["vel_rad_s"][1] == pytest.approx(math.radians(1.5))
    assert out["pos_rad"][1] == pytest.approx(math.radians(0.015))


def test_the_keys_are_identifys_keyword_arguments(csv_path):
    """``identify(**load_columns(...))`` has to just work."""
    rows = ["stamp,torque,speed,angle,other"]
    for i in range(400):
        t = i * 1e-3
        vel = math.sin(2 * math.pi * 3.0 * t)
        rows.append(f"{t:.6f},{0.4 * vel:.6f},{vel:.6f},{-math.cos(t):.6f},k")
    csv_path.write_text("\n".join(rows), encoding="utf-8")
    params = identify(
        **load_columns(str(csv_path), t="stamp", tau="torque", vel="speed", pos="angle")
    )
    assert params.coulomb_nm >= 0.0


def test_a_missing_column_says_which_one_and_what_was_available(csv_path):
    with pytest.raises(ValueError) as excinfo:
        load_columns(str(csv_path), t="stamp", tau="cmd_l", vel="speed")
    message = str(excinfo.value)
    assert "tau='cmd_l'" in message
    assert "Available columns" in message
    assert "torque" in message


def test_several_missing_columns_are_all_reported(csv_path):
    with pytest.raises(ValueError) as excinfo:
        load_columns(str(csv_path), t="nope", tau="nah", vel="speed", pos="neither")
    message = str(excinfo.value)
    assert "t='nope'" in message
    assert "tau='nah'" in message
    assert "pos='neither'" in message


def test_a_non_numeric_cell_names_the_column_and_the_row(csv_path):
    with pytest.raises(ValueError) as excinfo:
        load_columns(str(csv_path), t="stamp", tau="other", vel="speed")
    assert "'other'" in str(excinfo.value)
    assert "row 1" in str(excinfo.value)


def test_an_empty_cell_is_an_error(tmp_path):
    path = tmp_path / "gap.csv"
    path.write_text("a,b,c\n0,1,2\n1,,3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty cell"):
        load_columns(str(path), t="a", tau="b", vel="c")


def test_a_header_with_no_rows_is_an_error(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("a,b,c\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no data rows"):
        load_columns(str(path), t="a", tau="b", vel="c")


def test_a_file_with_no_header_is_an_error(tmp_path):
    path = tmp_path / "blank.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no header row"):
        load_columns(str(path), t="a", tau="b", vel="c")


def test_a_byte_order_mark_does_not_break_the_first_header(tmp_path):
    path = tmp_path / "bom.csv"
    path.write_text(SAMPLE, encoding="utf-8-sig")
    out = load_columns(str(path), t="stamp", tau="torque", vel="speed")
    assert out["t_s"][0] == 0.0


def test_an_unknown_velocity_unit_is_rejected(csv_path):
    with pytest.raises(ValueError, match="vel_unit"):
        load_columns(str(csv_path), t="stamp", tau="torque", vel="speed", vel_unit="rpm")  # type: ignore[arg-type]


@pytest.mark.parametrize("kwargs", [{"t": ""}, {"tau": None}, {"vel": 3}, {"pos": ""}])
def test_column_names_must_be_non_empty_strings(csv_path, kwargs):
    call = {"t": "stamp", "tau": "torque", "vel": "speed"} | kwargs
    with pytest.raises(ValueError):
        load_columns(str(csv_path), **call)


def test_a_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_columns(str(tmp_path / "nope.csv"), t="a", tau="b", vel="c")


def test_no_project_header_name_is_hard_coded():
    """The contract's red line: the library must not know any consumer's column names."""
    source = (Path(__file__).parents[1] / "src" / "sim2real_actuator").rglob("*.py")
    forbidden = ("cmd_l", "cmd_r", "ldps", "rdps", "ldeg", "rdeg", "host_t", "kpa")
    for path in source:
        text = path.read_text(encoding="utf-8")
        for name in forbidden:
            assert name not in text, f"{path.name} hard-codes the column name {name!r}"
