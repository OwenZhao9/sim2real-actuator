"""Single-joint actuator simulator.

:meth:`ActuatorSim.step` is pure arithmetic: no IO, no clock reads, no blocking.  It is
safe to call from a real-time control loop.  Instances are **not** thread safe and hold
no lock -- give each thread its own instance.
"""

from __future__ import annotations

import math

import numpy as np

from ._errors import Sim2RealActuatorError
from .params import ActuatorParams, SimState

__all__ = ["ActuatorSim"]


class ActuatorSim:
    """Integrate one geared joint under Coulomb + viscous friction, bias and backlash.

    The shaft obeys::

        inertia * accel = tau_cmd + external + bias
                          - viscous * vel - coulomb * sign(vel)

    While the joint stands still it stays still until the non-friction torque exceeds
    ``coulomb_nm`` (static friction holds it), which is what makes the simulated joint
    behave like a real gearbox instead of creeping on every tiny command.

    Args:
        p: Identified parameters.  ``inertia_kg_m2`` must be set -- integrating needs it.
        seed: Seed for this instance's private RNG.  See the note below.

    Raises:
        ValueError: If ``p.inertia_kg_m2`` is ``None`` or ``seed`` is not a valid seed.

    Note:
        ``step()`` in v0.1.0 is fully deterministic and consumes **nothing** from the
        private RNG -- ``seed`` changes no output today.  It is accepted (and the
        generator really is built) so that a per-instance reproducible stream exists for
        callers and for future stochastic effects.  The reproducibility guarantee that is
        live today is :func:`sim2real_actuator.randomize`.
    """

    __slots__ = (
        "_backlash_half",
        "_bias",
        "_coulomb",
        "_inv_inertia",
        "_moving",
        "_p",
        "_pos_motor",
        "_pos_out",
        "_rng",
        "_seed",
        "_tau_limit",
        "_vel",
        "_viscous",
    )

    def __init__(self, p: ActuatorParams, *, seed: int | None = None) -> None:
        if not isinstance(p, ActuatorParams):
            raise Sim2RealActuatorError(f"p must be an ActuatorParams, got {type(p).__name__}")
        if p.inertia_kg_m2 is None:
            raise Sim2RealActuatorError(
                "ActuatorSim needs p.inertia_kg_m2; it is None. Identify with "
                "fit_inertia=True, or set it explicitly: "
                "dataclasses.replace(p, inertia_kg_m2=...)"
            )
        try:
            self._rng = np.random.default_rng(seed)
        except (TypeError, ValueError) as exc:
            raise Sim2RealActuatorError(f"invalid seed {seed!r}: {exc}") from exc

        self._p = p
        self._seed = seed
        self._coulomb = float(p.coulomb_nm)
        self._viscous = float(p.viscous_nm_s_per_rad)
        self._bias = float(p.bias_nm)
        self._inv_inertia = 1.0 / float(p.inertia_kg_m2)
        self._backlash_half = 0.5 * float(p.backlash_rad)
        self._tau_limit = p.tau_limit_nm  # None or a positive float
        self._pos_motor = 0.0
        self._pos_out = 0.0
        self._vel = 0.0
        self._moving = False

    @property
    def params(self) -> ActuatorParams:
        """The parameters this instance was built with (read-only)."""
        return self._p

    def reset(self, pos_rad: float = 0.0, vel_rad_s: float = 0.0) -> None:
        """Place the joint at a known state and clear the backlash dead band.

        Args:
            pos_rad: Output position, rad.
            vel_rad_s: Output velocity, rad/s.

        Raises:
            ValueError: If either argument is not a finite real number.
        """
        try:
            pos = float(pos_rad)
            vel = float(vel_rad_s)
        except (TypeError, ValueError) as exc:
            raise Sim2RealActuatorError(f"reset() needs real numbers: {exc}") from exc
        if not (math.isfinite(pos) and math.isfinite(vel)):
            raise Sim2RealActuatorError(
                f"reset() needs finite values, got pos_rad={pos_rad!r}, vel_rad_s={vel_rad_s!r}"
            )
        self._pos_motor = pos
        self._pos_out = pos
        self._vel = vel
        self._moving = vel != 0.0

    def step(self, tau_cmd_nm: float, dt_s: float, *, external_nm: float = 0.0) -> SimState:
        """Advance the joint by ``dt_s`` under a commanded torque.

        Pure arithmetic -- no IO, no clock, no blocking, no locking.  Semi-implicit
        (symplectic) Euler integration.

        Args:
            tau_cmd_nm: Commanded actuator torque, Nm.  Clamped to ``tau_limit_nm``.
            dt_s: Time step, seconds, > 0.
            external_nm: Any other torque on the output this step (gravity, a load, a
                spring, a human), Nm.  Not subject to ``tau_limit_nm``.

        Returns:
            :class:`~sim2real_actuator.SimState` after the step.

        Raises:
            ValueError: Only for caller mistakes -- non-finite inputs or ``dt_s <= 0``.
                There is no IO, timeout or backend in this path, so there is nothing to
                degrade from; a bad number is a bug and is reported loudly.
        """
        dt = dt_s
        tau = tau_cmd_nm
        ext = external_nm
        try:
            ok = math.isfinite(dt) and dt > 0.0 and math.isfinite(tau) and math.isfinite(ext)
        except TypeError as exc:
            raise Sim2RealActuatorError(f"step() needs real numbers: {exc}") from exc
        if not ok:
            raise Sim2RealActuatorError(
                f"step() needs finite numbers with dt_s > 0, got tau_cmd_nm={tau_cmd_nm!r}, "
                f"dt_s={dt_s!r}, external_nm={external_nm!r}"
            )

        limit = self._tau_limit
        if limit is not None:
            if tau > limit:
                tau = limit
            elif tau < -limit:
                tau = -limit

        drive = tau + ext + self._bias
        coulomb = self._coulomb
        vel = self._vel

        if vel == 0.0:
            if -coulomb <= drive <= coulomb:
                # Static friction holds.  Nothing moves, nothing integrates.
                self._moving = False
                return SimState(self._pos_out, 0.0, tau, False)
            breakout = drive - (coulomb if drive > 0.0 else -coulomb)
            vel_new = breakout * self._inv_inertia * dt
            self._moving = True
        else:
            friction = (coulomb if vel > 0.0 else -coulomb) + self._viscous * vel
            vel_new = vel + (drive - friction) * self._inv_inertia * dt
            # Velocity tried to reverse inside this step: it can only really reverse if
            # the drive torque can also break static friction the other way.
            reversed_sign = vel_new != 0.0 and (vel_new > 0.0) != (vel > 0.0)
            if reversed_sign and -coulomb <= drive <= coulomb:
                self._moving = False
                self._vel = 0.0
                return SimState(self._pos_out, 0.0, tau, False)
            self._moving = vel_new != 0.0

        self._vel = vel_new
        self._pos_motor += vel_new * dt

        half = self._backlash_half
        if half > 0.0:
            slack = self._pos_motor - self._pos_out
            if slack > half:
                self._pos_out = self._pos_motor - half
            elif slack < -half:
                self._pos_out = self._pos_motor + half
        else:
            self._pos_out = self._pos_motor

        return SimState(self._pos_out, vel_new, tau, self._moving)
