"""The frozen data structures: validation and JSON round-trips (contract 0.2)."""

from __future__ import annotations

import dataclasses
import json

import pytest

from sim2real_actuator import (
    ActuatorParams,
    DomainRanges,
    Sim2RealActuatorError,
    SimState,
)

FULL = ActuatorParams(
    coulomb_nm=0.31,
    viscous_nm_s_per_rad=0.08,
    bias_nm=-0.19,
    backlash_rad=0.004,
    inertia_kg_m2=0.02,
    tau_limit_nm=1.5,
    source="unit test",
)


@pytest.mark.parametrize(
    "value",
    [
        FULL,
        ActuatorParams(coulomb_nm=0.0, viscous_nm_s_per_rad=0.0, bias_nm=0.0),
        SimState(1.25, -0.5, 0.33, True),
        SimState(0.0, 0.0, 0.0, False),
        DomainRanges(),
        DomainRanges(coulomb=(0.5, 2.0), viscous=(1.0, 1.0), bias=(-1.0, 1.0), backlash=(0.0, 0.1)),
    ],
)
def test_json_round_trip(value):
    blob = json.dumps(value.to_dict())
    assert type(value).from_dict(json.loads(blob)) == value


def test_frozen_dataclasses_cannot_be_mutated():
    with pytest.raises(dataclasses.FrozenInstanceError):
        FULL.coulomb_nm = 1.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        SimState(0.0, 0.0, 0.0, False).moving = True  # type: ignore[misc]


def test_library_error_is_a_value_error():
    assert issubclass(Sim2RealActuatorError, ValueError)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"coulomb_nm": -0.1, "viscous_nm_s_per_rad": 0.0, "bias_nm": 0.0},
        {"coulomb_nm": 0.0, "viscous_nm_s_per_rad": -1.0, "bias_nm": 0.0},
        {"coulomb_nm": 0.0, "viscous_nm_s_per_rad": 0.0, "bias_nm": float("nan")},
        {"coulomb_nm": float("inf"), "viscous_nm_s_per_rad": 0.0, "bias_nm": 0.0},
        {"coulomb_nm": 0.0, "viscous_nm_s_per_rad": 0.0, "bias_nm": 0.0, "backlash_rad": -1e-6},
        {"coulomb_nm": 0.0, "viscous_nm_s_per_rad": 0.0, "bias_nm": 0.0, "inertia_kg_m2": 0.0},
        {"coulomb_nm": 0.0, "viscous_nm_s_per_rad": 0.0, "bias_nm": 0.0, "tau_limit_nm": -1.0},
        {"coulomb_nm": 0.0, "viscous_nm_s_per_rad": 0.0, "bias_nm": 0.0, "source": 42},
        {"coulomb_nm": "heavy", "viscous_nm_s_per_rad": 0.0, "bias_nm": 0.0},
    ],
)
def test_construction_errors_are_value_errors(kwargs):
    with pytest.raises(ValueError):
        ActuatorParams(**kwargs)


def test_optional_fields_default_sensibly():
    params = ActuatorParams(coulomb_nm=0.1, viscous_nm_s_per_rad=0.2, bias_nm=0.3)
    assert params.backlash_rad == 0.0
    assert params.inertia_kg_m2 is None
    assert params.tau_limit_nm is None
    assert params.source == ""


def test_from_dict_rejects_unknown_keys():
    payload = FULL.to_dict() | {"stiffness_nm_per_rad": 3.0}
    with pytest.raises(ValueError, match="unknown key"):
        ActuatorParams.from_dict(payload)


def test_from_dict_reports_missing_required_keys():
    with pytest.raises(ValueError, match="missing key"):
        ActuatorParams.from_dict({"coulomb_nm": 0.1})
    with pytest.raises(ValueError, match="missing key"):
        SimState.from_dict({"pos_rad": 0.0})


def test_from_dict_rejects_non_dicts():
    for cls in (ActuatorParams, SimState, DomainRanges):
        with pytest.raises(ValueError, match="needs a dict"):
            cls.from_dict([1, 2, 3])  # type: ignore[arg-type]


def test_from_dict_fills_optional_fields():
    params = ActuatorParams.from_dict(
        {"coulomb_nm": 0.1, "viscous_nm_s_per_rad": 0.2, "bias_nm": 0.3}
    )
    assert params == ActuatorParams(coulomb_nm=0.1, viscous_nm_s_per_rad=0.2, bias_nm=0.3)
    assert DomainRanges.from_dict({}) == DomainRanges()


def test_to_dict_is_json_safe():
    for value in (FULL, SimState(0.0, 1.0, 2.0, True), DomainRanges()):
        json.dumps(value.to_dict())  # must not raise


@pytest.mark.parametrize(
    "kwargs",
    [
        {"coulomb": (1.5, 0.5)},
        {"coulomb": (-0.1, 1.0)},
        {"viscous": (-2.0, 1.0)},
        {"bias": (1.0, -1.0)},
        {"backlash": (0.0, float("nan"))},
        {"coulomb": 0.9},
        {"bias": (1.0, 2.0, 3.0)},
    ],
)
def test_domain_range_validation(kwargs):
    with pytest.raises(ValueError):
        DomainRanges(**kwargs)


def test_domain_ranges_normalise_lists_to_tuples():
    ranges = DomainRanges.from_dict({"coulomb": [0.5, 1.5]})
    assert ranges.coulomb == (0.5, 1.5)


def test_sim_state_does_not_validate_on_the_hot_path():
    """SimState is produced inside step(); it must stay allocation-cheap, not defensive."""
    state = SimState(float("inf"), float("nan"), 0.0, True)
    assert state.moving is True
