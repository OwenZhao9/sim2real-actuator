"""Domain randomization over :class:`~sim2real_actuator.ActuatorParams`."""

from __future__ import annotations

import numpy as np

from ._errors import Sim2RealActuatorError
from .params import ActuatorParams, DomainRanges

__all__ = ["randomize"]


def randomize(p: ActuatorParams, ranges: DomainRanges, n: int, seed: int) -> list[ActuatorParams]:
    """Draw ``n`` perturbed copies of ``p``.

    ``coulomb`` and ``viscous`` are scaled multiplicatively, ``bias`` and ``backlash``
    are shifted additively (``backlash_rad`` is clamped at 0).  ``inertia_kg_m2``,
    ``tau_limit_nm`` are carried over unchanged -- :class:`DomainRanges` has no field for
    them, and this library does not invent API beyond its contract.

    Args:
        p: The nominal parameters (usually the output of
            :func:`~sim2real_actuator.identify`).
        ranges: Per-parameter sampling ranges.
        n: How many parameter sets to draw, >= 0.
        seed: RNG seed.  The same ``(p, ranges, n, seed)`` always yields the exact same
            list, on any machine, and the global numpy RNG is never touched.

    Returns:
        A list of ``n`` :class:`~sim2real_actuator.ActuatorParams`.

    Raises:
        ValueError: If ``p`` / ``ranges`` have the wrong type, ``n`` is negative, or
            ``seed`` is not a valid numpy seed.
    """
    if not isinstance(p, ActuatorParams):
        raise Sim2RealActuatorError(f"p must be an ActuatorParams, got {type(p).__name__}")
    if not isinstance(ranges, DomainRanges):
        raise Sim2RealActuatorError(f"ranges must be a DomainRanges, got {type(ranges).__name__}")
    if isinstance(n, bool) or not isinstance(n, int):
        raise Sim2RealActuatorError(f"n must be an int, got {type(n).__name__}")
    if n < 0:
        raise Sim2RealActuatorError(f"n must be >= 0, got {n}")
    try:
        rng = np.random.default_rng(seed)
    except (TypeError, ValueError) as exc:
        raise Sim2RealActuatorError(f"invalid seed {seed!r}: {exc}") from exc
    if n == 0:
        return []

    coulomb = p.coulomb_nm * rng.uniform(ranges.coulomb[0], ranges.coulomb[1], n)
    viscous = p.viscous_nm_s_per_rad * rng.uniform(ranges.viscous[0], ranges.viscous[1], n)
    bias = p.bias_nm + rng.uniform(ranges.bias[0], ranges.bias[1], n)
    backlash = p.backlash_rad + rng.uniform(ranges.backlash[0], ranges.backlash[1], n)

    return [
        ActuatorParams(
            coulomb_nm=max(0.0, float(coulomb[i])),
            viscous_nm_s_per_rad=max(0.0, float(viscous[i])),
            bias_nm=float(bias[i]),
            backlash_rad=max(0.0, float(backlash[i])),
            inertia_kg_m2=p.inertia_kg_m2,
            tau_limit_nm=p.tau_limit_nm,
            source=f"{p.source}|randomize(seed={seed},i={i})",
        )
        for i in range(n)
    ]
