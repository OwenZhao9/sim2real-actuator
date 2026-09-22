from __future__ import annotations

import pytest

from sim2real_actuator import ActuatorParams


@pytest.fixture
def truth() -> ActuatorParams:
    return ActuatorParams(
        coulomb_nm=0.35,
        viscous_nm_s_per_rad=0.08,
        bias_nm=0.12,
        inertia_kg_m2=0.006,
        source="synthetic truth",
    )
