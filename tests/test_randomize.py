"""randomize(): reproducible domain randomization."""

from __future__ import annotations

import numpy as np
import pytest

from sim2real_actuator import ActuatorParams, DomainRanges, randomize

NOMINAL = ActuatorParams(
    coulomb_nm=0.30,
    viscous_nm_s_per_rad=0.10,
    bias_nm=0.22,
    backlash_rad=0.005,
    inertia_kg_m2=0.02,
    tau_limit_nm=1.5,
    source="nominal",
)


def test_same_seed_same_result():
    a = randomize(NOMINAL, DomainRanges(), 32, seed=1234)
    b = randomize(NOMINAL, DomainRanges(), 32, seed=1234)
    assert [p.to_dict() for p in a] == [p.to_dict() for p in b]


def test_different_seed_different_result():
    a = randomize(NOMINAL, DomainRanges(), 32, seed=1)
    b = randomize(NOMINAL, DomainRanges(), 32, seed=2)
    assert [p.coulomb_nm for p in a] != [p.coulomb_nm for p in b]


def test_the_global_numpy_rng_is_never_touched():
    np.random.seed(0)
    before = np.random.random()
    np.random.seed(0)
    randomize(NOMINAL, DomainRanges(), 100, seed=99)
    assert np.random.random() == before


def test_samples_stay_inside_the_requested_ranges():
    ranges = DomainRanges(
        coulomb=(0.8, 1.25), viscous=(0.5, 2.0), bias=(-0.05, 0.05), backlash=(0.0, 0.02)
    )
    for p in randomize(NOMINAL, ranges, 500, seed=7):
        assert 0.8 * NOMINAL.coulomb_nm <= p.coulomb_nm <= 1.25 * NOMINAL.coulomb_nm
        assert (
            0.5 * NOMINAL.viscous_nm_s_per_rad
            <= p.viscous_nm_s_per_rad
            <= 2.0 * NOMINAL.viscous_nm_s_per_rad
        )
        assert NOMINAL.bias_nm - 0.05 <= p.bias_nm <= NOMINAL.bias_nm + 0.05
        assert NOMINAL.backlash_rad <= p.backlash_rad <= NOMINAL.backlash_rad + 0.02


def test_unrandomised_fields_are_carried_over():
    for p in randomize(NOMINAL, DomainRanges(), 10, seed=3):
        assert p.inertia_kg_m2 == NOMINAL.inertia_kg_m2
        assert p.tau_limit_nm == NOMINAL.tau_limit_nm
        assert p.source.startswith("nominal|randomize(seed=3,")


def test_a_degenerate_range_reproduces_the_nominal_parameters():
    ranges = DomainRanges(
        coulomb=(1.0, 1.0), viscous=(1.0, 1.0), bias=(0.0, 0.0), backlash=(0.0, 0.0)
    )
    for p in randomize(NOMINAL, ranges, 5, seed=0):
        assert p.coulomb_nm == pytest.approx(NOMINAL.coulomb_nm)
        assert p.viscous_nm_s_per_rad == pytest.approx(NOMINAL.viscous_nm_s_per_rad)
        assert p.bias_nm == pytest.approx(NOMINAL.bias_nm)
        assert p.backlash_rad == pytest.approx(NOMINAL.backlash_rad)


def test_backlash_never_goes_negative():
    ranges = DomainRanges(backlash=(-1.0, -0.5))
    for p in randomize(NOMINAL, ranges, 50, seed=11):
        assert p.backlash_rad >= 0.0


def test_zero_draws_returns_an_empty_list():
    assert randomize(NOMINAL, DomainRanges(), 0, seed=0) == []


@pytest.mark.parametrize(
    ("p", "ranges", "n", "seed"),
    [
        ("not params", DomainRanges(), 3, 0),
        (NOMINAL, "not ranges", 3, 0),
        (NOMINAL, DomainRanges(), -1, 0),
        (NOMINAL, DomainRanges(), 1.5, 0),
        (NOMINAL, DomainRanges(), True, 0),
        (NOMINAL, DomainRanges(), 3, "not a seed"),
    ],
)
def test_bad_arguments_are_rejected(p, ranges, n, seed):
    with pytest.raises(ValueError):
        randomize(p, ranges, n, seed)
