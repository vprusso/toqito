"""Test random_psd_operator."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from toqito.matrix_props import is_positive_semidefinite
from toqito.rand import random_psd_operator


@pytest.mark.parametrize(
    "dim, is_real, distribution",
    [
        # Test with a matrix of dimension 2, real, uniform.
        (2, True, "uniform"),
        # Test with a matrix of dimension 4, complex, uniform.
        (4, False, "uniform"),
        # Test with a matrix of dimension 5, complex, uniform.
        (5, False, "uniform"),
        # Test with a matrix of dimension 10, real, uniform.
        (10, True, "uniform"),
        # Test with a matrix of dimension 4, complex, wishart.
        (4, False, "wishart"),
        # Test with a matrix of dimension 4, real, wishart.
        (4, True, "wishart"),
    ],
)
def test_random_psd_operator(dim, is_real, distribution):
    """Test for random_psd_operator function."""
    rand_psd_operator = random_psd_operator(dim, is_real, distribution=distribution)
    assert_equal(rand_psd_operator.shape, (dim, dim))
    assert is_positive_semidefinite(rand_psd_operator)


@pytest.mark.parametrize(
    "dim, is_real, seed, distribution, expected_mat",
    [
        # Uniform real, dim=2, seed=13.
        (
            2,
            True,
            13,
            "uniform",
            np.array(
                [
                    [1.0778147, 0.52948168],
                    [0.52948168, 0.6943802],
                ]
            ),
        ),
        # Uniform complex, dim=3, seed=42.
        (
            3,
            False,
            42,
            "uniform",
            np.array(
                [
                    [0.95967267 + 1.23259516e-32j, 0.64147554 + 2.12105374e-01j, 0.58409076 - 4.68163042e-03j],
                    [0.64147554 - 2.12105374e-01j, 0.77763615 - 1.38777878e-17j, 0.12628199 + 2.28264771e-01j],
                    [0.58409076 + 4.68163042e-03j, 0.12628199 - 2.28264771e-01j, 1.07906147 + 0.00000000e00j],
                ]
            ),
        ),
        # Uniform real, dim=5, seed=13.
        (
            5,
            True,
            13,
            "uniform",
            np.array(
                [
                    [1.10423147, 0.58541728, 0.23882546, 0.38725184, 0.47981462],
                    [0.58541728, 1.1071704, 0.65708784, 0.59531392, 0.53463283],
                    [0.23882546, 0.65708784, 0.80971882, 0.31269181, 0.34134362],
                    [0.38725184, 0.59531392, 0.31269181, 0.85758043, 0.54105916],
                    [0.47981462, 0.53463283, 0.34134362, 0.54105916, 1.22308851],
                ]
            ),
        ),
        # Wishart real, dim=3, seed=7.
        (
            3,
            True,
            7,
            "wishart",
            np.array(
                [
                    [0.79677259, 0.48589897, 0.85321202],
                    [0.48589897, 2.09215132, -0.29068742],
                    [0.85321202, -0.29068742, 1.30078171],
                ]
            ),
        ),
        # Wishart complex, dim=3, seed=7.
        (
            3,
            False,
            7,
            "wishart",
            np.array(
                [
                    [0.83816019 + 4.20667209e-18j, -0.42537849 - 2.49304337e-02j, 0.15525367 - 1.37819291e-01j],
                    [-0.42537849 + 2.49304337e-02j, 2.50239004 + 2.84697454e-17j, 0.26324124 + 9.71713807e-01j],
                    [0.15525367 + 1.37819291e-01j, 0.26324124 - 9.71713807e-01j, 0.81920895 + 7.23372601e-18j],
                ]
            ),
        ),
    ],
)
def test_random_psd_operator_with_seed(dim, is_real, seed, distribution, expected_mat):
    """Test that random_psd_operator function returns the expected output when seeded."""
    matrix = random_psd_operator(dim, is_real, seed, distribution=distribution)
    assert_allclose(matrix, expected_mat, rtol=1e-6)


@pytest.mark.parametrize(
    "dim, distribution",
    [
        # Invalid dim type.
        ("4", "uniform"),
        # Negative dim.
        (-2, "uniform"),
        # Zero dim.
        (0, "uniform"),
    ],
)
def test_random_psd_operator_invalid_dim(dim, distribution):
    """Test that invalid dim raises ValueError."""
    with pytest.raises(ValueError):
        random_psd_operator(dim, distribution=distribution)


def test_random_psd_operator_invalid_distribution():
    """Test that invalid distribution raises ValueError."""
    with pytest.raises(ValueError):
        random_psd_operator(4, distribution="invalid")


@pytest.mark.parametrize(
    "scale, num_degrees, match",
    [
        # scale shape mismatch.
        (np.eye(3), None, "scale must be a 4x4 matrix"),
        # Non-PSD scale matrix.
        (
            np.array(
                [
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, -1.0],
                ]
            ),
            None,
            "positive semidefinite",
        ),
        # num_degrees zero.
        (None, 0, "num_degrees must be a positive integer"),
        # num_degrees negative.
        (None, -3, "num_degrees must be a positive integer"),
    ],
)
def test_random_psd_operator_wishart_invalid_params(scale, num_degrees, match):
    """Test that invalid Wishart parameters raise ValueError."""
    with pytest.raises(ValueError, match=match):
        random_psd_operator(4, distribution="wishart", scale=scale, num_degrees=num_degrees)


def test_random_psd_operator_uniform_ignores_wishart_params():
    """Test that passing scale or num_degrees with distribution='uniform' raises a warning."""
    with pytest.warns(UserWarning, match="scale and num_degrees are ignored"):
        random_psd_operator(4, distribution="uniform", scale=np.eye(4), num_degrees=5)


def test_random_psd_operator_wishart_rank_deficient_warning():
    """Test that num_degrees < dim raises a UserWarning about rank deficiency."""
    with pytest.warns(UserWarning, match="rank-deficient"):
        random_psd_operator(4, distribution="wishart", num_degrees=2)
