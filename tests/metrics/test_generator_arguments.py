"""Default arguments and argument validation of the metric generators."""

import numpy as np
import pytest

from warpica.metrics import (
    metric_get_alcubierre,
    metric_get_alcubierre_comoving,
    metric_get_lentz,
    metric_get_lentz_comoving,
    metric_get_minkowski,
    metric_get_modified_time,
    metric_get_modified_time_comoving,
    metric_get_schwarzschild,
    metric_get_van_den_broeck,
    metric_get_van_den_broeck_comoving,
    metric_get_warp_shell_comoving,
)
from warpica.metrics.lentz.metric_get_lentz import get_warp_factor_by_region
from warpica.metrics.lentz.metric_get_lentz_comoving import (
    get_warp_factor_by_region as get_warp_factor_by_region_comoving,
)
from warpica.units import G, c

GRID = np.array([1, 5, 5, 5])
CENTRE = np.array([1, 2.8, 2.7, 2.6])

# Each entry builds a metric from a grid_scale argument.
GENERATORS = {
    "minkowski": lambda s: metric_get_minkowski(GRID, s),
    "alcubierre": lambda s: metric_get_alcubierre(GRID, CENTRE, 0.2, 1, 1, s),
    "alcubierre_comoving": lambda s: metric_get_alcubierre_comoving(
        GRID, CENTRE, 0.2, 1, 1, s
    ),
    "lentz": lambda s: metric_get_lentz(GRID, CENTRE, 0.2, 1.0, s),
    "lentz_comoving": lambda s: metric_get_lentz_comoving(GRID, CENTRE, 0.2, 1.0, s),
    "modified_time": lambda s: metric_get_modified_time(GRID, CENTRE, 0.2, 1, 1, 2, s),
    "modified_time_comoving": lambda s: metric_get_modified_time_comoving(
        GRID, CENTRE, 0.2, 1, 1, 2, s
    ),
    "van_den_broeck": lambda s: metric_get_van_den_broeck(
        GRID, CENTRE, 0.2, 1, 1, 2, 1, 0.1, s
    ),
    "van_den_broeck_comoving": lambda s: metric_get_van_den_broeck_comoving(
        GRID, CENTRE, 0.2, 1, 1, 2, 1, 0.1, s
    ),
    "schwarzschild": lambda s: metric_get_schwarzschild(GRID, CENTRE, 0.05, s),
    "warp_shell": lambda s: metric_get_warp_shell_comoving(
        GRID, CENTRE, 2e26, 0.5, 1.5, 0.1, 1, 1, 0.2, True, s
    ),
}


@pytest.mark.parametrize("name", list(GENERATORS))
def test_the_grid_scale_defaults_to_unit_spacing(name):
    default = GENERATORS[name](None)
    explicit = GENERATORS[name](np.ones(4))
    np.testing.assert_array_equal(
        np.array(default["tensor"]), np.array(explicit["tensor"])
    )
    np.testing.assert_array_equal(default["scaling"], np.ones(4))


# ---- Lentz ------------------------------------------------------------------


@pytest.mark.parametrize(
    "generator",
    [metric_get_lentz, metric_get_lentz_comoving],
    ids=["lentz", "comoving"],
)
@pytest.mark.parametrize("container", [np.array, list], ids=["array", "list"])
def test_the_lentz_scale_defaults_to_the_largest_dimension_over_seven(
    generator, container
):
    grid = container([1, 7, 14, 7])
    centre = np.array([1, 4.1, 7.2, 4.3])
    default = generator(grid, centre, 0.2, None, None)
    explicit = generator(grid, centre, 0.2, 2.0, None)
    np.testing.assert_array_equal(
        np.array(default["tensor"]), np.array(explicit["tensor"])
    )


# One point per region of the Lentz template, with the values read from
# upstream's getWarpFactorByRegion (size scale 1). The third entry is the
# region's description; the last column is the sign convention for y < 0.
REGIONS = [
    ((1.5, 0.0), (-2, 0), "outer front, y near 0"),
    ((1.5, 1.0), (-1, 1), "outer front, diagonal band"),
    ((0.5, 1.0), (0, 1), "inner front"),
    ((0.5, 2.0), (-0.5, 0.5), "inner front, outer band"),
    ((-0.5, 2.0), (0.5, 0.5), "rear, outer band"),
    ((-0.5, 0.6), (1, 0), "rear, inner"),
    ((0.0, 0.5), (1, 0), "central strip"),
    ((5.0, 0.0), (0, 0), "outside every region"),
]
REGION_FUNCTIONS = [get_warp_factor_by_region, get_warp_factor_by_region_comoving]


@pytest.mark.parametrize("region", REGION_FUNCTIONS, ids=["lentz", "comoving"])
@pytest.mark.parametrize(
    ("point", "expected", "label"), REGIONS, ids=[r[2] for r in REGIONS]
)
def test_each_lentz_region_has_the_upstream_factors(region, point, expected, label):
    assert region(*point, 1) == expected


@pytest.mark.parametrize("region", REGION_FUNCTIONS, ids=["lentz", "comoving"])
@pytest.mark.parametrize(
    ("point", "expected", "label"), REGIONS, ids=[r[2] for r in REGIONS]
)
def test_the_lentz_template_is_antisymmetric_in_y(region, point, expected, label):
    x, y = point
    factor_x, factor_y = region(x, -y, 1)
    assert factor_x == expected[0]
    assert factor_y == -expected[1]


@pytest.mark.parametrize("region", REGION_FUNCTIONS, ids=["lentz", "comoving"])
@pytest.mark.parametrize(
    ("point", "expected", "label"), REGIONS, ids=[r[2] for r in REGIONS]
)
def test_the_lentz_template_scales_with_its_size(region, point, expected, label):
    x, y = point
    assert region(3.7 * x, 3.7 * y, 3.7) == expected


# ---- Schwarzschild ----------------------------------------------------------


def test_schwarzschild_needs_a_single_time_slice():
    with pytest.raises(ValueError, match="one time slice"):
        metric_get_schwarzschild(np.array([2, 5, 5, 5]), CENTRE, 0.05, np.ones(4))


def test_schwarzschild_radius_must_not_be_negative():
    with pytest.raises(ValueError, match="nonnegative"):
        metric_get_schwarzschild(GRID, CENTRE, -0.1, np.ones(4))


# ---- warp shell -------------------------------------------------------------


def shell(**overrides):
    arguments = {
        "m": 2e26,
        "R_1": 0.5,
        "R_2": 1.5,
        "R_buff": 0.1,
        "sigma": 1,
        "smooth_factor": 1,
        "v_warp": 0.2,
        "do_warp": True,
    }
    arguments.update(overrides)
    return metric_get_warp_shell_comoving(
        np.array([1, 7, 7, 7]),
        np.array([1, 4.1, 4.2, 3.9]),
        grid_scaling=np.ones(4),
        **arguments,
    )


def mass_at_fraction_of_limit(fraction, R_2=1.5):
    """Mass whose Schwarzschild radius is `fraction` of the 8 R_2 / 9 limit."""
    return fraction * (8 * R_2 / 9) * c() ** 2 / (2 * G())


@pytest.mark.parametrize(
    "overrides",
    [
        {"m": -1.0},
        {"R_1": -0.1},
        {"R_1": 1.5},
        {"R_1": 2.0},
        {"R_buff": -0.1},
        {"R_buff": 0.5},
        {"R_buff": 0.6},
    ],
    ids=[
        "negative mass",
        "negative R_1",
        "R_1 = R_2",
        "R_1 > R_2",
        "negative buffer",
        "buffer fills the shell",
        "buffer exceeds the shell",
    ],
)
def test_shell_geometry_is_validated(overrides):
    with pytest.raises(ValueError, match="nonnegative mass"):
        shell(**overrides)


@pytest.mark.parametrize("overrides", [{"smooth_factor": -1}, {"sigma": -1}])
def test_shell_smoothing_and_sharpness_must_not_be_negative(overrides):
    with pytest.raises(ValueError, match="must be nonnegative"):
        shell(**overrides)


def test_shell_must_respect_the_compactness_limit():
    shell(m=mass_at_fraction_of_limit(0.99))
    with pytest.raises(ValueError, match="compactness limit"):
        shell(m=mass_at_fraction_of_limit(1.01))


def test_smoothing_cannot_create_a_coordinate_horizon():
    """A shell near the limit, smoothed over a huge window, spills mass inward."""
    kwargs = {"R_1": 0.0, "R_buff": 0.0, "sigma": 0, "v_warp": 0.0, "do_warp": False}
    shell(m=mass_at_fraction_of_limit(0.9), smooth_factor=1, **kwargs)
    with pytest.raises(ValueError, match="coordinate horizon"):
        shell(m=mass_at_fraction_of_limit(0.9), smooth_factor=20000, **kwargs)
