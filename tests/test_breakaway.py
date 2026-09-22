"""breakaway(): the two-measurement split of static friction and bias."""

from __future__ import annotations

import pytest

from sim2real_actuator import ActuatorParams, ActuatorSim, breakaway


def test_symmetric_thresholds_mean_no_bias():
    static, bias = breakaway(0.40, 0.40)
    assert static == pytest.approx(0.40)
    assert bias == pytest.approx(0.0)


def test_the_contract_formula():
    static, bias = breakaway(0.08, 0.52)
    assert (static, bias) == (pytest.approx(0.30), pytest.approx(0.22))


@pytest.mark.parametrize(
    ("up", "down"),
    [(0.0, 0.0), (0.1, 0.9), (1.5, 0.2), (0.277, 0.479), (3.0, 3.0)],
)
def test_inverse_of_up_equals_static_minus_bias(up, down):
    static, bias = breakaway(up, down)
    assert static - bias == pytest.approx(up)
    assert static + bias == pytest.approx(down)


def test_bias_is_positive_when_the_joint_starts_more_easily_upwards():
    _, bias = breakaway(0.10, 0.50)
    assert bias > 0.0


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf")])
def test_invalid_arguments_are_rejected(bad):
    with pytest.raises(ValueError):
        breakaway(bad, 0.3)
    with pytest.raises(ValueError):
        breakaway(0.3, bad)


def test_agrees_with_the_simulator(  # the two halves of the library must tell one story
):
    """Sweep a ramp through ActuatorSim, read the break-away torques back out."""
    truth = ActuatorParams(
        coulomb_nm=0.30, viscous_nm_s_per_rad=0.05, bias_nm=0.22, inertia_kg_m2=0.01
    )
    measured = {}
    for direction in (+1.0, -1.0):
        sim = ActuatorSim(truth)
        sim.reset()
        torque = 0.0
        while torque < 5.0:
            torque += 1e-4
            if sim.step(direction * torque, 1e-4).moving:
                measured[direction] = torque
                break
    static, bias = breakaway(measured[+1.0], measured[-1.0])
    assert static == pytest.approx(truth.coulomb_nm, abs=2e-3)
    assert bias == pytest.approx(truth.bias_nm, abs=2e-3)


def test_agrees_with_the_real_reference_numbers():
    """The quoted hip-exo reference values invert to the recorded ramp thresholds."""
    # right joint, as quoted for the real device: static 0.40 Nm, bias +0.11 Nm
    static, bias = breakaway(0.29, 0.51)
    assert static == pytest.approx(0.40, abs=0.01)
    assert bias == pytest.approx(0.11, abs=0.01)
