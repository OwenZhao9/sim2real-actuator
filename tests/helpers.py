"""Shared test helpers (importable because pytest puts ``tests/`` on ``sys.path``)."""

from __future__ import annotations

import math

from sim2real_actuator import ActuatorParams, ActuatorSim


def sweep(
    params: ActuatorParams,
    *,
    dt: float = 1e-3,
    steps: int = 40_000,
    amplitude: float = 1.4,
    seed: int | None = 7,
) -> dict[str, list[float]]:
    """Run a rich multi-sine excitation through the simulator and log it like a recorder.

    The excitation has to swing the joint through *both* directions, otherwise
    ``coulomb_nm`` and ``bias_nm`` are not separable.
    """
    sim = ActuatorSim(params, seed=seed)
    sim.reset()
    t_s: list[float] = []
    tau_nm: list[float] = []
    vel_rad_s: list[float] = []
    pos_rad: list[float] = []
    t = 0.0
    for _ in range(steps):
        tau = (
            amplitude * math.sin(2 * math.pi * 0.7 * t)
            + 0.43 * amplitude * math.sin(2 * math.pi * 2.3 * t + 0.4)
            + 0.25 * amplitude * math.sin(2 * math.pi * 5.1 * t + 1.1)
        )
        state = sim.step(tau, dt)
        t += dt
        t_s.append(t)
        tau_nm.append(tau)
        vel_rad_s.append(state.vel_rad_s)
        pos_rad.append(state.pos_rad)
    return {"t_s": t_s, "tau_nm": tau_nm, "vel_rad_s": vel_rad_s, "pos_rad": pos_rad}
