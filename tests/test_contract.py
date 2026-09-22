"""Guards on the frozen contract itself: the public surface and the house rules."""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

import pytest

import sim2real_actuator

SOURCE_DIR = Path(sim2real_actuator.__file__).parent

#: Exactly what the contract freezes, plus the library error base class that the
#: common section of the contract requires every library to define.
CONTRACT_SYMBOLS = {
    "ActuatorParams",
    "SimState",
    "DomainRanges",
    "identify",
    "breakaway",
    "ActuatorSim",
    "randomize",
    "load_columns",
}


def test_every_contract_symbol_is_importable():
    for name in CONTRACT_SYMBOLS:
        assert hasattr(sim2real_actuator, name), name


def test_nothing_extra_is_exported():
    extra = set(sim2real_actuator.__all__) - CONTRACT_SYMBOLS
    assert extra == {"Sim2RealActuatorError", "__version__"}


def test_the_contracts_import_line_works():
    exec(  # the contract's own import line, verbatim
        "from sim2real_actuator import ("
        "ActuatorParams, SimState, DomainRanges, "
        "identify, breakaway, ActuatorSim, randomize, load_columns)",
        {},
    )


def _render(func) -> str:
    """Render a signature the way the contract writes it: names, kinds and defaults."""
    parts: list[str] = []
    seen_keyword_only = False
    for parameter in inspect.signature(func).parameters.values():
        if parameter.kind is parameter.KEYWORD_ONLY and not seen_keyword_only:
            parts.append("*")
            seen_keyword_only = True
        if parameter.default is parameter.empty:
            parts.append(parameter.name)
        else:
            parts.append(f"{parameter.name}={parameter.default!r}")
    return "(" + ", ".join(parts) + ")"


@pytest.mark.parametrize(
    ("func", "signature"),
    [
        (
            sim2real_actuator.identify,
            "(*, t_s, tau_nm, vel_rad_s, pos_rad=None, fit_inertia=True)",
        ),
        (sim2real_actuator.breakaway, "(up_nm, down_nm)"),
        (sim2real_actuator.randomize, "(p, ranges, n, seed)"),
        (
            sim2real_actuator.load_columns,
            "(path, *, t, tau, vel, pos=None, vel_unit='rad/s')",
        ),
    ],
)
def test_signatures_match_the_contract(func, signature):
    assert _render(func) == signature


def test_actuator_sim_methods_match_the_contract():
    assert list(inspect.signature(sim2real_actuator.ActuatorSim.__init__).parameters) == [
        "self",
        "p",
        "seed",
    ]
    assert list(inspect.signature(sim2real_actuator.ActuatorSim.reset).parameters) == [
        "self",
        "pos_rad",
        "vel_rad_s",
    ]
    assert list(inspect.signature(sim2real_actuator.ActuatorSim.step).parameters) == [
        "self",
        "tau_cmd_nm",
        "dt_s",
        "external_nm",
    ]


def test_dataclass_fields_match_the_contract():
    import dataclasses

    assert [f.name for f in dataclasses.fields(sim2real_actuator.ActuatorParams)] == [
        "coulomb_nm",
        "viscous_nm_s_per_rad",
        "bias_nm",
        "backlash_rad",
        "inertia_kg_m2",
        "tau_limit_nm",
        "source",
    ]
    assert [f.name for f in dataclasses.fields(sim2real_actuator.SimState)] == [
        "pos_rad",
        "vel_rad_s",
        "tau_applied_nm",
        "moving",
    ]
    assert [f.name for f in dataclasses.fields(sim2real_actuator.DomainRanges)] == [
        "coulomb",
        "viscous",
        "bias",
        "backlash",
    ]


def test_domain_range_defaults_match_the_contract():
    ranges = sim2real_actuator.DomainRanges()
    assert ranges.coulomb == (0.8, 1.25)
    assert ranges.viscous == (0.8, 1.25)
    assert ranges.bias == (-0.05, 0.05)
    assert ranges.backlash == (0.0, 0.02)


def _library_sources() -> list[Path]:
    return sorted(SOURCE_DIR.rglob("*.py"))


def test_the_library_never_prints():
    """Contract 0.2: library code uses logging, never print()."""
    for path in _library_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            is_print = (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "print"
            )
            assert not is_print, f"{path.name}:{node.lineno} calls print()"


def test_the_library_never_reads_a_clock():
    """Contract 0.2: time is passed in by the caller, never sampled inside."""
    for path in _library_sources():
        text = path.read_text(encoding="utf-8")
        for banned in ("import time", "time.time(", "time.monotonic(", "datetime.now("):
            assert banned not in text, f"{path.name} uses {banned}"


def test_the_library_never_locks():
    """Contract 0.3: instances are not thread safe and must not pay for a lock."""
    for path in _library_sources():
        text = path.read_text(encoding="utf-8")
        for banned in ("import threading", "Lock(", "RLock(", "Semaphore("):
            assert banned not in text, f"{path.name} uses {banned}"


def test_the_library_imports_no_hardware_and_no_sibling_library():
    """Contract 0.1 / 6.3: no serial ports, no cross-imports between the four libraries."""
    banned_modules = {
        "serial",
        "pyserial",
        "usb",
        "smbus",
        "RPi",
        "fly_reflex",
        "jev_decide",
        "evomap_genes",
    }
    for path in _library_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or "").split(".")[0]]
            else:
                continue
            assert not banned_modules & set(names), f"{path.name} imports {names}"


def test_the_only_third_party_dependency_is_numpy():
    stdlib_or_self = {"__future__", "sim2real_actuator", ""}
    third_party: set[str] = set()
    for path in _library_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                third_party |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                third_party.add((node.module or "").split(".")[0])
    third_party -= stdlib_or_self
    non_stdlib = {
        name
        for name in third_party
        if importlib.util.find_spec(name) is not None
        and "site-packages" in (importlib.util.find_spec(name).origin or "")
    }
    assert non_stdlib <= {"numpy"}, non_stdlib


def test_the_package_ships_a_py_typed_marker():
    assert (SOURCE_DIR / "py.typed").is_file()


def test_every_public_symbol_is_annotated_and_documented():
    for name in sorted(CONTRACT_SYMBOLS):
        obj = getattr(sim2real_actuator, name)
        assert obj.__doc__, f"{name} has no docstring"
        target = obj.__init__ if inspect.isclass(obj) else obj
        hints = inspect.get_annotations(target)
        expected = [p for p in inspect.signature(target).parameters if p != "self"]
        missing = [p for p in expected if p not in hints]
        assert not missing, f"{name} is missing annotations for {missing}"
        if not inspect.isclass(obj):
            assert "return" in hints, f"{name} has no return annotation"
