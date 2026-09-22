"""sim2real-actuator -- actuator identification, simulation and domain randomization.

A minimal, dependency-light subset of `BAM <https://github.com/Rhoban/bam>`_: identify a
geared actuator's friction and bias from a recording, replay it in a single-joint
simulator, and shake the parameters for domain randomization.  Nothing here is tied to a
physics engine, a robot or a fieldbus.

Sign convention (used everywhere)::

    inertia * accel = tau_cmd + external + bias_nm
                      - viscous_nm_s_per_rad * vel - coulomb_nm * sign(vel)

Thread safety: no object in this package is thread safe and none of them take a lock.
One instance, one thread.
"""

from __future__ import annotations

from ._errors import Sim2RealActuatorError
from .identify import breakaway, identify
from .loaders import load_columns
from .params import ActuatorParams, DomainRanges, SimState
from .randomize import randomize
from .sim import ActuatorSim

__version__ = "0.1.0"

__all__ = [
    "ActuatorParams",
    "ActuatorSim",
    "DomainRanges",
    "Sim2RealActuatorError",
    "SimState",
    "__version__",
    "breakaway",
    "identify",
    "load_columns",
    "randomize",
]
