"""identify(): the closed loop that the whole library stands on."""

from __future__ import annotations

import math

import pytest

from helpers import sweep
from sim2real_actuator import ActuatorParams, identify

CLOSED_LOOP_TOLERANCE = 0.05  # the contract's bar: every parameter within 5 %


def relative_error(got: float, want: float) -> float:
    return abs(got - want) / abs(want)


def test_synthetic_closed_loop_recovers_every_parameter(truth):
    """Known parameters -> ActuatorSim trajectory -> identify -> the same parameters."""
    log = sweep(truth)
    got = identify(**log)

    errors = {
        "coulomb_nm": relative_error(got.coulomb_nm, truth.coulomb_nm),
        "viscous_nm_s_per_rad": relative_error(
            got.viscous_nm_s_per_rad, truth.viscous_nm_s_per_rad
        ),
        "bias_nm": relative_error(got.bias_nm, truth.bias_nm),
        "inertia_kg_m2": relative_error(got.inertia_kg_m2, truth.inertia_kg_m2),
    }
    worst = max(errors.values())
    assert worst < CLOSED_LOOP_TOLERANCE, errors


@pytest.mark.parametrize(
    "params",
    [
        ActuatorParams(
            coulomb_nm=0.05, viscous_nm_s_per_rad=0.30, bias_nm=-0.22, inertia_kg_m2=0.02
        ),
        ActuatorParams(
            coulomb_nm=0.60, viscous_nm_s_per_rad=0.01, bias_nm=0.00, inertia_kg_m2=0.002
        ),
        ActuatorParams(
            coulomb_nm=0.20, viscous_nm_s_per_rad=0.15, bias_nm=0.40, inertia_kg_m2=0.05
        ),
    ],
)
def test_closed_loop_over_a_range_of_actuators(params):
    got = identify(**sweep(params, amplitude=2.0))
    assert relative_error(got.coulomb_nm, params.coulomb_nm) < CLOSED_LOOP_TOLERANCE
    assert (
        relative_error(got.viscous_nm_s_per_rad, params.viscous_nm_s_per_rad)
        < CLOSED_LOOP_TOLERANCE
    )
    assert relative_error(got.inertia_kg_m2, params.inertia_kg_m2) < CLOSED_LOOP_TOLERANCE
    if params.bias_nm:
        assert relative_error(got.bias_nm, params.bias_nm) < CLOSED_LOOP_TOLERANCE
    else:
        assert abs(got.bias_nm) < 0.02


def test_identification_is_deterministic(truth):
    log = sweep(truth)
    assert identify(**log) == identify(**log)


def test_position_column_is_optional(truth):
    log = sweep(truth)
    without = identify(t_s=log["t_s"], tau_nm=log["tau_nm"], vel_rad_s=log["vel_rad_s"])
    with_pos = identify(**log)
    assert relative_error(without.coulomb_nm, with_pos.coulomb_nm) < 0.02


def test_fit_inertia_false_drops_the_inertia_term(truth):
    log = sweep(truth)
    got = identify(**log, fit_inertia=False)
    assert got.inertia_kg_m2 is None
    # Friction is still recovered, just with the acceleration term folded into the noise.
    assert relative_error(got.coulomb_nm, truth.coulomb_nm) < 0.20


def test_velocity_rail_samples_are_dropped(truth):
    """A fixed-point clamp on the velocity column must not poison the fit."""
    log = sweep(truth)
    clean = identify(**log)

    rail = max(abs(v) for v in log["vel_rad_s"]) * 0.6
    railed = [max(-rail, min(rail, v)) for v in log["vel_rad_s"]]
    assert sum(1 for v in railed if abs(v) == rail) > 100, "the fixture must actually saturate"

    got = identify(t_s=log["t_s"], tau_nm=log["tau_nm"], vel_rad_s=railed, pos_rad=log["pos_rad"])
    assert "dropped_saturated=" in got.source
    dropped = int(got.source.split("dropped_saturated=")[1].split(",")[0])
    assert dropped > 100
    assert relative_error(got.coulomb_nm, clean.coulomb_nm) < 0.25


def test_standing_still_samples_are_dropped(truth):
    """A long dead segment where sign(vel) is meaningless must not move the answer."""
    log = sweep(truth)
    clean = identify(**log)

    t0 = log["t_s"][-1]
    padded = {
        "t_s": log["t_s"] + [t0 + 1e-3 * (i + 1) for i in range(4000)],
        "tau_nm": log["tau_nm"] + [0.9] * 4000,
        "vel_rad_s": log["vel_rad_s"] + [0.0] * 4000,
        "pos_rad": log["pos_rad"] + [log["pos_rad"][-1]] * 4000,
    }
    got = identify(**padded)
    assert relative_error(got.coulomb_nm, clean.coulomb_nm) < 0.02
    assert relative_error(got.bias_nm, clean.bias_nm) < 0.02


def test_frozen_encoder_counts_as_standing_still(truth):
    """pos_rad sharpens the static mask: a non-ticking encoder is not moving."""
    log = sweep(truth)
    n = 3000
    t0 = log["t_s"][-1]
    # Velocity noise big enough to pass the speed threshold, but the encoder never ticks.
    noisy = [0.02 * math.sin(i) for i in range(n)]
    padded = {
        "t_s": log["t_s"] + [t0 + 1e-3 * (i + 1) for i in range(n)],
        "tau_nm": log["tau_nm"] + [1.2] * n,
        "vel_rad_s": log["vel_rad_s"] + noisy,
        "pos_rad": log["pos_rad"] + [log["pos_rad"][-1]] * n,
    }
    with_pos = identify(**padded)
    assert relative_error(with_pos.coulomb_nm, truth.coulomb_nm) < CLOSED_LOOP_TOLERANCE


def test_single_direction_data_is_rejected_with_a_useful_message(truth):
    log = sweep(truth)
    one_way = {**log, "vel_rad_s": [abs(v) for v in log["vel_rad_s"]]}
    with pytest.raises(ValueError, match="both directions"):
        identify(**one_way)


def test_length_mismatch_is_rejected(truth):
    log = sweep(truth, steps=100)
    with pytest.raises(ValueError, match="same length"):
        identify(t_s=log["t_s"][:-1], tau_nm=log["tau_nm"], vel_rad_s=log["vel_rad_s"])


def test_too_few_samples_is_rejected():
    with pytest.raises(ValueError, match="at least 8 samples"):
        identify(t_s=[0.0, 1.0], tau_nm=[0.0, 1.0], vel_rad_s=[0.0, 1.0])


def test_time_must_not_run_backwards():
    n = 20
    with pytest.raises(ValueError, match="non-decreasing"):
        identify(
            t_s=[float(n - i) for i in range(n)],
            tau_nm=[0.1] * n,
            vel_rad_s=[0.1 * (-1) ** i for i in range(n)],
        )


def test_frozen_clock_is_rejected():
    n = 20
    with pytest.raises(ValueError, match="never advances"):
        identify(t_s=[0.0] * n, tau_nm=[0.1] * n, vel_rad_s=[0.1 * (-1) ** i for i in range(n)])


def test_all_static_input_is_rejected():
    n = 200
    with pytest.raises(ValueError, match="usable samples"):
        identify(t_s=[i * 1e-3 for i in range(n)], tau_nm=[0.3] * n, vel_rad_s=[0.0] * n)


def test_non_numeric_input_is_rejected():
    with pytest.raises(ValueError):
        identify(t_s=[0.0, "x", 2.0], tau_nm=[0.0] * 3, vel_rad_s=[0.0] * 3)


def test_backlash_and_torque_limit_are_not_estimated(truth):
    got = identify(**sweep(truth))
    assert got.backlash_rad == 0.0
    assert got.tau_limit_nm is None
    assert "identify(" in got.source
