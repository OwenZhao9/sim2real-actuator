"""ActuatorSim: the physics, the determinism, and the real-time promises."""

from __future__ import annotations

import builtins
import dataclasses
import socket
import time

import pytest

from sim2real_actuator import ActuatorParams, ActuatorSim, SimState

FRICTIONLESS = ActuatorParams(
    coulomb_nm=0.0, viscous_nm_s_per_rad=0.0, bias_nm=0.0, inertia_kg_m2=0.01
)


def test_inertia_is_required():
    params = ActuatorParams(coulomb_nm=0.1, viscous_nm_s_per_rad=0.0, bias_nm=0.0)
    with pytest.raises(ValueError, match="inertia_kg_m2"):
        ActuatorSim(params)


def test_wrong_type_is_rejected():
    with pytest.raises(ValueError, match="ActuatorParams"):
        ActuatorSim({"coulomb_nm": 0.1})  # type: ignore[arg-type]


def test_frictionless_joint_follows_newton():
    sim = ActuatorSim(FRICTIONLESS)
    sim.reset()
    state = SimState(0.0, 0.0, 0.0, False)
    for _ in range(1000):
        state = sim.step(0.05, 1e-3)
    # a = tau / J = 5 rad/s^2 for 1.0 s
    assert state.vel_rad_s == pytest.approx(5.0, rel=1e-9)
    assert state.pos_rad == pytest.approx(2.5, rel=1e-2)


def test_static_friction_holds_the_joint_still():
    params = ActuatorParams(
        coulomb_nm=0.30, viscous_nm_s_per_rad=0.0, bias_nm=0.0, inertia_kg_m2=0.01
    )
    sim = ActuatorSim(params)
    sim.reset()
    for _ in range(500):
        state = sim.step(0.29, 1e-3)
        assert state.vel_rad_s == 0.0
        assert state.pos_rad == 0.0
        assert state.moving is False
    assert sim.step(0.31, 1e-3).moving is True


def test_bias_shifts_where_the_joint_breaks_away():
    params = ActuatorParams(
        coulomb_nm=0.30, viscous_nm_s_per_rad=0.0, bias_nm=0.22, inertia_kg_m2=0.01
    )
    sim = ActuatorSim(params)
    sim.reset()
    # up needs only coulomb - bias = 0.08 Nm ...
    assert sim.step(0.079, 1e-4).moving is False
    sim.reset()
    assert sim.step(0.081, 1e-4).moving is True
    # ... while down needs coulomb + bias = 0.52 Nm
    sim.reset()
    assert sim.step(-0.51, 1e-4).moving is False
    sim.reset()
    assert sim.step(-0.53, 1e-4).moving is True


def test_coulomb_friction_brings_a_spinning_joint_to_a_stop():
    params = ActuatorParams(
        coulomb_nm=0.20, viscous_nm_s_per_rad=0.0, bias_nm=0.0, inertia_kg_m2=0.01
    )
    sim = ActuatorSim(params)
    sim.reset(0.0, 5.0)
    state = SimState(0.0, 5.0, 0.0, True)
    for _ in range(5000):
        state = sim.step(0.0, 1e-3)
        if not state.moving:
            break
    assert state.moving is False
    assert state.vel_rad_s == 0.0
    # decel = 0.2 / 0.01 = 20 rad/s^2, so 5 rad/s dies in 0.25 s
    assert state.pos_rad == pytest.approx(0.625, rel=0.05)


def test_viscous_friction_produces_a_terminal_velocity():
    params = ActuatorParams(
        coulomb_nm=0.0, viscous_nm_s_per_rad=0.1, bias_nm=0.0, inertia_kg_m2=0.001
    )
    sim = ActuatorSim(params)
    sim.reset()
    for _ in range(200_000):
        state = sim.step(0.5, 1e-4)
    assert state.vel_rad_s == pytest.approx(5.0, rel=1e-3)  # tau / b


def test_external_torque_is_added_to_the_command():
    sim = ActuatorSim(FRICTIONLESS)
    sim.reset()
    for _ in range(100):
        cancelled = sim.step(0.5, 1e-3, external_nm=-0.5)
    assert cancelled.vel_rad_s == pytest.approx(0.0, abs=1e-15)
    sim.reset()
    for _ in range(100):
        doubled = sim.step(0.5, 1e-3, external_nm=0.5)
    assert doubled.vel_rad_s == pytest.approx(10.0, rel=1e-9)  # 1.0 Nm / 0.01 kg m^2 * 0.1 s


def test_torque_limit_clamps_the_command_but_not_the_external_torque():
    params = dataclasses.replace(FRICTIONLESS, tau_limit_nm=0.2)
    sim = ActuatorSim(params)
    sim.reset()
    state = sim.step(10.0, 1e-3)
    assert state.tau_applied_nm == pytest.approx(0.2)
    sim.reset()
    assert sim.step(-10.0, 1e-3).tau_applied_nm == pytest.approx(-0.2)
    sim.reset()
    state = sim.step(10.0, 1e-3, external_nm=5.0)
    assert state.tau_applied_nm == pytest.approx(0.2)
    assert state.vel_rad_s == pytest.approx(5.2 / 0.01 * 1e-3)


def test_backlash_is_a_dead_band_on_the_reported_position():
    """The output lags the motor by half the play, and flips side on a reversal."""
    play = 0.02
    params = ActuatorParams(
        coulomb_nm=0.0,
        viscous_nm_s_per_rad=0.0,
        bias_nm=0.0,
        backlash_rad=play,
        inertia_kg_m2=0.001,
    )
    lashed = ActuatorSim(params)
    tight = ActuatorSim(dataclasses.replace(params, backlash_rad=0.0))
    lashed.reset()
    tight.reset()

    # Drive forward until the play is fully taken up.
    for _ in range(1000):
        forward_lashed = lashed.step(0.01, 1e-3)
        forward_tight = tight.step(0.01, 1e-3)
    assert forward_tight.pos_rad > play
    assert forward_lashed.pos_rad == pytest.approx(forward_tight.pos_rad - play / 2, abs=1e-12)

    # Reverse hard enough that the motor actually travels backwards past the play.
    for _ in range(4000):
        back_lashed = lashed.step(-0.01, 1e-3)
        back_tight = tight.step(-0.01, 1e-3)
    assert back_tight.pos_rad < forward_tight.pos_rad - play
    assert back_lashed.pos_rad == pytest.approx(back_tight.pos_rad + play / 2, abs=1e-12)

    # Velocity is the motor's; only the reported position carries the dead band.
    assert back_lashed.vel_rad_s == pytest.approx(back_tight.vel_rad_s, abs=1e-12)


def test_zero_backlash_leaves_the_position_untouched():
    sim = ActuatorSim(FRICTIONLESS)
    sim.reset(1.25, 0.0)
    assert sim.step(0.0, 1e-3).pos_rad == 1.25


def test_reset_puts_the_joint_back_exactly():
    sim = ActuatorSim(FRICTIONLESS)
    sim.reset(0.3, -1.5)
    first = sim.step(0.1, 1e-3)
    sim.reset(0.3, -1.5)
    assert sim.step(0.1, 1e-3) == first


def test_replaying_the_same_commands_gives_bit_identical_results():
    params = ActuatorParams(
        coulomb_nm=0.2, viscous_nm_s_per_rad=0.05, bias_nm=0.1, inertia_kg_m2=0.004
    )
    commands = [(0.1 * i % 1.3 - 0.6, 1e-3) for i in range(5000)]

    def run(seed):
        sim = ActuatorSim(params, seed=seed)
        sim.reset()
        return [sim.step(tau, dt).to_dict() for tau, dt in commands]

    assert run(0) == run(0)
    # seed changes nothing in v0.1.0: step() is deterministic and consumes no randomness.
    assert run(0) == run(12345)
    assert run(None) == run(0)


def test_two_instances_do_not_share_state():
    a = ActuatorSim(FRICTIONLESS)
    b = ActuatorSim(FRICTIONLESS)
    a.reset(1.0, 0.0)
    b.reset(-1.0, 0.0)
    assert a.step(0.0, 1e-3).pos_rad == 1.0
    assert b.step(0.0, 1e-3).pos_rad == -1.0


def test_params_property_returns_what_was_passed_in():
    sim = ActuatorSim(FRICTIONLESS)
    assert sim.params == FRICTIONLESS


@pytest.mark.parametrize(
    ("tau", "dt", "external"),
    [
        (0.0, 0.0, 0.0),
        (0.0, -1e-3, 0.0),
        (float("nan"), 1e-3, 0.0),
        (0.0, float("inf"), 0.0),
        (0.0, 1e-3, float("nan")),
        (float("inf"), 1e-3, 0.0),
    ],
)
def test_bad_numbers_are_rejected_loudly(tau, dt, external):
    sim = ActuatorSim(FRICTIONLESS)
    sim.reset()
    with pytest.raises(ValueError):
        sim.step(tau, dt, external_nm=external)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_reset_rejects_bad_numbers(bad):
    sim = ActuatorSim(FRICTIONLESS)
    with pytest.raises(ValueError):
        sim.reset(bad, 0.0)
    with pytest.raises(ValueError):
        sim.reset(0.0, bad)


def test_step_does_no_io_and_never_reads_a_clock(monkeypatch):
    """The real-time promise, enforced rather than asserted in prose."""

    def forbidden(*_args, **_kwargs):
        raise AssertionError("step() touched the outside world")

    sim = ActuatorSim(
        ActuatorParams(
            coulomb_nm=0.2,
            viscous_nm_s_per_rad=0.05,
            bias_nm=0.1,
            backlash_rad=0.01,
            tau_limit_nm=2.0,
            inertia_kg_m2=0.004,
        )
    )
    sim.reset()
    monkeypatch.setattr(time, "time", forbidden)
    monkeypatch.setattr(time, "monotonic", forbidden)
    monkeypatch.setattr(time, "perf_counter", forbidden)
    monkeypatch.setattr(time, "sleep", forbidden)
    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    for i in range(2000):
        sim.step(0.5 - 0.001 * i, 1e-3, external_nm=0.05)


def test_step_is_fast_enough_for_a_real_time_loop():
    """Generous bound -- this guards against an accidental O(n) or allocation blow-up."""
    sim = ActuatorSim(
        ActuatorParams(
            coulomb_nm=0.2,
            viscous_nm_s_per_rad=0.05,
            bias_nm=0.1,
            backlash_rad=0.01,
            inertia_kg_m2=0.004,
        )
    )
    sim.reset()
    iterations = 200_000
    start = time.perf_counter()
    for i in range(iterations):
        sim.step(0.6 if i % 500 < 250 else -0.6, 1e-4)
    per_call_us = (time.perf_counter() - start) / iterations * 1e6
    assert per_call_us < 50.0, f"{per_call_us:.2f} us per step()"
