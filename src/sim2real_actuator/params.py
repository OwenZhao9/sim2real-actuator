"""Frozen, JSON round-trippable data structures.

Sign convention (one convention, used by every function in this package)
-----------------------------------------------------------------------
The shaft obeys::

    inertia * accel = tau_cmd + external + bias_nm
                      - viscous_nm_s_per_rad * vel
                      - coulomb_nm * sign(vel)

so ``bias_nm`` is a *constant torque the actuator itself produces*, positive when it
pushes along the positive direction -- exactly what ``ActuatorParams.bias_nm`` is
documented as ("power-on base torque, signed, positive = along the positive
direction") and exactly the quantity :func:`sim2real_actuator.breakaway` returns.

Under this convention the two break-away torques are
``up = coulomb_nm - bias_nm`` and ``down = coulomb_nm + bias_nm`` (both magnitudes),
which inverts to ``breakaway(up, down) == ((up + down) / 2, (down - up) / 2)``.

The least-squares model quoted in the contract, ``tau ~ J*a + b*w + c*sign(w) + bias``,
solves for the *extra torque a caller must supply*, i.e. its intercept is
``-bias_nm``.  :func:`sim2real_actuator.identify` fits that regression and negates the
intercept before storing it, so ``identify`` and ``breakaway`` agree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._errors import Sim2RealActuatorError
from ._util import check_finite, check_non_negative, check_positive_or_none, unknown_keys

__all__ = ["ActuatorParams", "DomainRanges", "SimState"]


@dataclass(frozen=True)
class ActuatorParams:
    """Friction / bias model of one geared actuator.

    Args:
        coulomb_nm: Coulomb friction (constant drag opposing the motion), Nm, >= 0.
        viscous_nm_s_per_rad: Viscous friction coefficient, Nm per (rad/s), >= 0.
        bias_nm: Power-on base torque, signed; positive = pushes along +, Nm.
        backlash_rad: Total gear backlash (the whole dead band, not the half), rad, >= 0.
        inertia_kg_m2: Reflected inertia at the output, kg*m^2, > 0 or None if unknown.
            :class:`~sim2real_actuator.ActuatorSim` requires it.
        tau_limit_nm: Torque the drive can actually produce, Nm, > 0 or None for no limit.
        source: Free-form provenance string ("which measurement did this come from").
    """

    coulomb_nm: float
    viscous_nm_s_per_rad: float
    bias_nm: float
    backlash_rad: float = 0.0
    inertia_kg_m2: float | None = None
    tau_limit_nm: float | None = None
    source: str = ""

    _KEYS = frozenset(
        {
            "coulomb_nm",
            "viscous_nm_s_per_rad",
            "bias_nm",
            "backlash_rad",
            "inertia_kg_m2",
            "tau_limit_nm",
            "source",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "coulomb_nm", check_non_negative("coulomb_nm", self.coulomb_nm))
        object.__setattr__(
            self,
            "viscous_nm_s_per_rad",
            check_non_negative("viscous_nm_s_per_rad", self.viscous_nm_s_per_rad),
        )
        object.__setattr__(self, "bias_nm", check_finite("bias_nm", self.bias_nm))
        object.__setattr__(
            self, "backlash_rad", check_non_negative("backlash_rad", self.backlash_rad)
        )
        object.__setattr__(
            self, "inertia_kg_m2", check_positive_or_none("inertia_kg_m2", self.inertia_kg_m2)
        )
        object.__setattr__(
            self, "tau_limit_nm", check_positive_or_none("tau_limit_nm", self.tau_limit_nm)
        )
        if not isinstance(self.source, str):
            raise Sim2RealActuatorError(f"source must be str, got {type(self.source).__name__}")

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict; round-trips through :meth:`from_dict`."""
        return {
            "coulomb_nm": self.coulomb_nm,
            "viscous_nm_s_per_rad": self.viscous_nm_s_per_rad,
            "bias_nm": self.bias_nm,
            "backlash_rad": self.backlash_rad,
            "inertia_kg_m2": self.inertia_kg_m2,
            "tau_limit_nm": self.tau_limit_nm,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ActuatorParams:
        """Inverse of :meth:`to_dict`.  Unknown keys raise :class:`ValueError`."""
        if not isinstance(d, dict):
            raise Sim2RealActuatorError(f"from_dict() needs a dict, got {type(d).__name__}")
        unknown_keys(cls.__name__, d, cls._KEYS)
        missing = sorted({"coulomb_nm", "viscous_nm_s_per_rad", "bias_nm"} - set(d))
        if missing:
            raise Sim2RealActuatorError(f"{cls.__name__}.from_dict() missing key(s): {missing}")
        return cls(
            coulomb_nm=d["coulomb_nm"],
            viscous_nm_s_per_rad=d["viscous_nm_s_per_rad"],
            bias_nm=d["bias_nm"],
            backlash_rad=d.get("backlash_rad", 0.0),
            inertia_kg_m2=d.get("inertia_kg_m2"),
            tau_limit_nm=d.get("tau_limit_nm"),
            source=d.get("source", ""),
        )


@dataclass(frozen=True)
class SimState:
    """One simulation sample.

    Args:
        pos_rad: Output-side position, rad (includes the backlash dead band).
        vel_rad_s: Output-side velocity, rad/s.
        tau_applied_nm: Torque the drive actually applied this step -- the command after
            ``tau_limit_nm`` clamping.  It excludes ``external_nm`` and ``bias_nm``.
        moving: ``False`` while static friction holds the joint at zero velocity.

    This is a plain value record produced by :meth:`~sim2real_actuator.ActuatorSim.step`
    on the real-time path, so it deliberately performs **no** validation.
    """

    pos_rad: float
    vel_rad_s: float
    tau_applied_nm: float
    moving: bool

    _KEYS = frozenset({"pos_rad", "vel_rad_s", "tau_applied_nm", "moving"})

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict; round-trips through :meth:`from_dict`."""
        return {
            "pos_rad": self.pos_rad,
            "vel_rad_s": self.vel_rad_s,
            "tau_applied_nm": self.tau_applied_nm,
            "moving": self.moving,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SimState:
        """Inverse of :meth:`to_dict`.  Unknown keys raise :class:`ValueError`."""
        if not isinstance(d, dict):
            raise Sim2RealActuatorError(f"from_dict() needs a dict, got {type(d).__name__}")
        unknown_keys(cls.__name__, d, cls._KEYS)
        missing = sorted(cls._KEYS - set(d))
        if missing:
            raise Sim2RealActuatorError(f"{cls.__name__}.from_dict() missing key(s): {missing}")
        return cls(
            pos_rad=float(d["pos_rad"]),
            vel_rad_s=float(d["vel_rad_s"]),
            tau_applied_nm=float(d["tau_applied_nm"]),
            moving=bool(d["moving"]),
        )


def _check_range(name: str, value: Any, *, multiplicative: bool) -> tuple[float, float]:
    try:
        lo, hi = value
    except (TypeError, ValueError) as exc:
        raise Sim2RealActuatorError(f"{name} must be a (lo, hi) pair, got {value!r}") from exc
    lo = check_finite(f"{name}[0]", lo)
    hi = check_finite(f"{name}[1]", hi)
    if lo > hi:
        raise Sim2RealActuatorError(f"{name} needs lo <= hi, got ({lo}, {hi})")
    if multiplicative and lo < 0.0:
        raise Sim2RealActuatorError(f"{name} is a multiplicative range and needs lo >= 0, got {lo}")
    return (lo, hi)


@dataclass(frozen=True)
class DomainRanges:
    """Per-parameter sampling ranges for :func:`sim2real_actuator.randomize`.

    Args:
        coulomb: Multiplicative range applied to ``coulomb_nm`` (dimensionless, >= 0).
        viscous: Multiplicative range applied to ``viscous_nm_s_per_rad`` (>= 0).
        bias: Additive range applied to ``bias_nm``, Nm.
        backlash: Additive range applied to ``backlash_rad``, rad.
    """

    coulomb: tuple[float, float] = (0.8, 1.25)
    viscous: tuple[float, float] = (0.8, 1.25)
    bias: tuple[float, float] = (-0.05, 0.05)
    backlash: tuple[float, float] = (0.0, 0.02)

    _KEYS = frozenset({"coulomb", "viscous", "bias", "backlash"})

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "coulomb", _check_range("coulomb", self.coulomb, multiplicative=True)
        )
        object.__setattr__(
            self, "viscous", _check_range("viscous", self.viscous, multiplicative=True)
        )
        object.__setattr__(self, "bias", _check_range("bias", self.bias, multiplicative=False))
        object.__setattr__(
            self, "backlash", _check_range("backlash", self.backlash, multiplicative=False)
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict (tuples become lists); round-trips through :meth:`from_dict`."""
        return {
            "coulomb": list(self.coulomb),
            "viscous": list(self.viscous),
            "bias": list(self.bias),
            "backlash": list(self.backlash),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DomainRanges:
        """Inverse of :meth:`to_dict`.  Unknown keys raise :class:`ValueError`."""
        if not isinstance(d, dict):
            raise Sim2RealActuatorError(f"from_dict() needs a dict, got {type(d).__name__}")
        unknown_keys(cls.__name__, d, cls._KEYS)
        defaults = cls()
        return cls(
            coulomb=tuple(d.get("coulomb", defaults.coulomb)),  # type: ignore[arg-type]
            viscous=tuple(d.get("viscous", defaults.viscous)),  # type: ignore[arg-type]
            bias=tuple(d.get("bias", defaults.bias)),  # type: ignore[arg-type]
            backlash=tuple(d.get("backlash", defaults.backlash)),  # type: ignore[arg-type]
        )
