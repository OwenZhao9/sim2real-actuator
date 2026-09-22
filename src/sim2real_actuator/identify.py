"""Least-squares friction / bias identification from a recorded motion."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from ._errors import Sim2RealActuatorError
from ._util import check_non_negative, to_float_array
from .params import ActuatorParams

__all__ = ["breakaway", "identify"]

_log = logging.getLogger(__name__)

#: A velocity rail (fixed-point clamp) is assumed when the extreme |vel| repeats at least
#: this many times.  Unit-agnostic on purpose -- no project-specific constant is baked in.
_RAIL_MIN_REPEATS = 3

#: Samples slower than ``_STATIC_FRAC * percentile(|vel|, 99.5)`` count as "standing still":
#: ``sign(vel)`` is meaningless there and the torque carries no kinetic-friction information.
_STATIC_FRAC = 0.02
_STATIC_FLOOR = 1e-9

#: Both directions of travel are needed to separate ``coulomb_nm`` from ``bias_nm``.
_MIN_PER_DIRECTION = 3


def breakaway(up_nm: float, down_nm: float) -> tuple[float, float]:
    """Split two break-away torque magnitudes into (static friction, bias).

    ``up_nm`` and ``down_nm`` are the *magnitudes* of the smallest commanded torque that
    just starts motion in the positive and the negative direction.  With a constant
    internal torque ``bias`` pushing along ``+``::

        up   = static - bias
        down = static + bias

    Args:
        up_nm: Break-away torque magnitude in the positive direction, Nm, >= 0.
        down_nm: Break-away torque magnitude in the negative direction, Nm, >= 0.

    Returns:
        ``(static_friction_nm, bias_nm)`` = ``((up + down) / 2, (down - up) / 2)``.
        ``bias_nm`` follows the same sign convention as
        :attr:`~sim2real_actuator.ActuatorParams.bias_nm`.

    Raises:
        ValueError: If either argument is negative or not finite.
    """
    up = check_non_negative("up_nm", up_nm)
    down = check_non_negative("down_nm", down_nm)
    return ((up + down) / 2.0, (down - up) / 2.0)


def _rail_mask(vel: np.ndarray, finite: np.ndarray) -> np.ndarray:
    """Mark saturated velocity samples (and their immediate neighbours)."""
    mask = np.zeros(vel.shape, dtype=bool)
    if not finite.any():
        return mask
    mag = np.abs(vel)
    peak = float(mag[finite].max())
    if peak <= 0.0:
        return mask
    # A fixed-point clamp emits bit-identical values, so compare almost exactly:
    # a smooth signal will not repeat its own maximum to 1e-12 three times over.
    at_peak = finite & (np.abs(mag - peak) <= 1e-12 * peak)
    if int(at_peak.sum()) < _RAIL_MIN_REPEATS:
        return mask
    # The derivative of a railed sample is corrupted on both sides, so dilate by one.
    mask |= at_peak
    mask[1:] |= at_peak[:-1]
    mask[:-1] |= at_peak[1:]
    return mask


def _static_mask(vel: np.ndarray, usable: np.ndarray, pos: np.ndarray | None) -> np.ndarray:
    """Mark samples that are standing still (``sign(vel)`` undefined)."""
    mag = np.abs(vel)
    ref = float(np.percentile(mag[usable], 99.5)) if usable.any() else 0.0
    threshold = max(_STATIC_FLOOR, _STATIC_FRAC * ref)
    mask = mag <= threshold
    if pos is not None and pos.size >= 3:
        # A quantised encoder that has not ticked is standing still, whatever the noisy
        # velocity column says.  Compare against both neighbours.
        frozen = np.zeros(pos.shape, dtype=bool)
        frozen[1:-1] = (pos[1:-1] == pos[:-2]) & (pos[1:-1] == pos[2:])
        mask |= frozen
    return mask


def _nonneg_lstsq(
    design: np.ndarray, target: np.ndarray, non_negative: Sequence[bool]
) -> np.ndarray:
    """Least squares with a few columns pinned to >= 0 (projection + refit)."""
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    active = np.ones(design.shape[1], dtype=bool)
    for _ in range(design.shape[1]):
        bad = [
            i for i in range(design.shape[1]) if active[i] and non_negative[i] and solution[i] < 0.0
        ]
        if not bad:
            break
        active[bad] = False
        solution = np.zeros(design.shape[1], dtype=np.float64)
        if not active.any():
            break
        partial, *_ = np.linalg.lstsq(design[:, active], target, rcond=None)
        solution[active] = partial
    return solution


def identify(
    *,
    t_s: Sequence[float],
    tau_nm: Sequence[float],
    vel_rad_s: Sequence[float],
    pos_rad: Sequence[float] | None = None,
    fit_inertia: bool = True,
) -> ActuatorParams:
    """Fit ``tau ~ J*accel + b*vel + c*sign(vel) + beta`` by least squares.

    Velocity-rail samples (a fixed-point clamp that repeats its extreme value) and
    standing-still segments are dropped automatically before the fit.

    Args:
        t_s: Sample timestamps, seconds, non-decreasing.
        tau_nm: Torque applied at each sample, Nm.
        vel_rad_s: Output velocity at each sample, rad/s.
        pos_rad: Optional output position, rad.  Only used to sharpen the
            standing-still mask (an encoder that has not ticked is not moving).
        fit_inertia: Fit the inertia term too.  When ``False`` the acceleration column is
            dropped and ``inertia_kg_m2`` is left ``None``.

    Returns:
        :class:`~sim2real_actuator.ActuatorParams`.  ``bias_nm`` is ``-beta`` -- the
        regression's intercept is the *extra torque the caller must supply*, while
        ``bias_nm`` is the torque the actuator itself produces (see
        :mod:`sim2real_actuator.params` for the single sign convention).
        ``backlash_rad`` stays ``0.0`` and ``tau_limit_nm`` stays ``None``: neither is
        estimated by this function.

    Raises:
        ValueError: If the sequences differ in length, are too short, contain no usable
            samples, or only ever move in one direction (``coulomb_nm`` and ``bias_nm``
            are not separable from single-direction data -- use :func:`breakaway`).
    """
    t = to_float_array("t_s", t_s)
    tau = to_float_array("tau_nm", tau_nm)
    vel = to_float_array("vel_rad_s", vel_rad_s)
    pos = to_float_array("pos_rad", pos_rad) if pos_rad is not None else None

    lengths = {"t_s": t.size, "tau_nm": tau.size, "vel_rad_s": vel.size}
    if pos is not None:
        lengths["pos_rad"] = pos.size
    if len(set(lengths.values())) != 1:
        raise Sim2RealActuatorError(f"all sequences must have the same length, got {lengths}")
    n = t.size
    if n < 8:
        raise Sim2RealActuatorError(f"need at least 8 samples to identify, got {n}")

    finite = np.isfinite(t) & np.isfinite(tau) & np.isfinite(vel)
    if pos is not None:
        finite &= np.isfinite(pos)
    if not finite.any():
        raise Sim2RealActuatorError("no finite samples in the input")

    dt = np.diff(t)
    if np.any(dt[np.isfinite(dt)] < 0.0):
        raise Sim2RealActuatorError("t_s must be non-decreasing")
    if not np.any(dt[np.isfinite(dt)] > 0.0):
        raise Sim2RealActuatorError("t_s never advances; cannot differentiate velocity")

    rails = _rail_mask(vel, finite)
    usable = finite & ~rails
    static = _static_mask(vel, usable, pos)
    keep = usable & ~static

    # Acceleration by central difference on the (possibly non-uniform) time grid.
    # np.gradient needs strictly increasing x, so fall back to a safe uniform grid when
    # the recorder emitted duplicate timestamps.
    if np.all(dt[np.isfinite(dt)] > 0.0) and np.all(np.isfinite(t)):
        accel = np.gradient(vel, t)
    else:
        step = float(np.median(dt[np.isfinite(dt) & (dt > 0.0)]))
        accel = np.gradient(vel, step)
        _log.warning("duplicate/non-finite timestamps; using the median step %.6g s", step)
    keep &= np.isfinite(accel)

    kept = int(keep.sum())
    n_cols = 4 if fit_inertia else 3
    if kept < n_cols + 1:
        raise Sim2RealActuatorError(
            f"only {kept} usable samples after dropping {int(rails.sum())} saturated and "
            f"{int((static & usable).sum())} standing-still samples; need at least {n_cols + 1}"
        )

    sgn = np.sign(vel[keep])
    n_pos = int((sgn > 0).sum())
    n_neg = int((sgn < 0).sum())
    if n_pos < _MIN_PER_DIRECTION or n_neg < _MIN_PER_DIRECTION:
        raise Sim2RealActuatorError(
            f"need motion in both directions to separate coulomb_nm from bias_nm "
            f"(got {n_pos} positive and {n_neg} negative samples, "
            f"at least {_MIN_PER_DIRECTION} each). Use breakaway() for one-way data."
        )

    ones = np.ones(kept, dtype=np.float64)
    if fit_inertia:
        design = np.column_stack((accel[keep], vel[keep], sgn, ones))
        non_negative = (True, True, True, False)
    else:
        design = np.column_stack((vel[keep], sgn, ones))
        non_negative = (True, True, False)

    solution = _nonneg_lstsq(design, tau[keep], non_negative)

    if fit_inertia:
        inertia, viscous, coulomb, intercept = (float(x) for x in solution)
        inertia_out: float | None = inertia if inertia > 0.0 else None
        if inertia_out is None:
            _log.warning("inertia fit was non-positive; reporting inertia_kg_m2=None")
    else:
        viscous, coulomb, intercept = (float(x) for x in solution)
        inertia_out = None

    residual = tau[keep] - design @ solution
    rms = float(np.sqrt(np.mean(residual**2))) if kept else float("nan")
    source = (
        f"identify(n={kept}/{n}, fit_inertia={fit_inertia}, "
        f"dropped_saturated={int(rails.sum())}, dropped_static={int((static & usable).sum())}, "
        f"rms_nm={rms:.4g})"
    )
    _log.debug("identify: %s", source)

    return ActuatorParams(
        coulomb_nm=max(0.0, coulomb),
        viscous_nm_s_per_rad=max(0.0, viscous),
        bias_nm=-intercept,
        backlash_rad=0.0,
        inertia_kg_m2=inertia_out,
        tau_limit_nm=None,
        source=source,
    )
