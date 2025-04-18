"""Test random_psd_operator."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from toqito.matrix_props import is_positive_semidefinite
from toqito.rand import random_psd_operator


@pytest.mark.parametrize(
    "dim, is_real",
    [
        # Test with a matrix of dimension 2.
        (2, True),
        # Test with a matrix of dimension 4.
        (4, False),
        # Test with a matrix of dimension 5.
        (5, False),
        # Test with a matrix of dimension 10.
        (10, True),
    ],
)
def test_random_psd_operator(dim, is_real):
    """Test for random_psd_operator function."""
    # Generate a random positive semidefinite operator.
    rand_psd_operator = random_psd_operator(dim, is_real)

    # Ensure the matrix has the correct shape.
    assert_equal(rand_psd_operator.shape, (dim, dim))

    # Check if the matrix is positive semidefinite.
    assert is_positive_semidefinite(rand_psd_operator)


@pytest.mark.parametrize(
    "dim, is_real, seed, expected_mat",
    [
        (
            2,
            True,
            13,
            np.array(
                [
                    [1.0778147, 0.52948168],
                    [0.52948168, 0.6943802],
                ]
            ),
        ),
        (
            3,
            False,
            42,
            np.array(
                [
                    [0.95967267 + 1.23259516e-32j, 0.64147554 + 2.12105374e-01j, 0.58409076 - 4.68163042e-03j],
                    [0.64147554 - 2.12105374e-01j, 0.77763615 - 1.38777878e-17j, 0.12628199 + 2.28264771e-01j],
                    [0.58409076 + 4.68163042e-03j, 0.12628199 - 2.28264771e-01j, 1.07906147 + 0.00000000e00j],
                ]
            ),
        ),
        (
            5,
            True,
            13,
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
    ],
)
def test_random_psd_operator_with_seed(dim, is_real, seed, expected_mat):
    """Test that random_psd_operator function returns the expected output when seeded."""
    matrix = random_psd_operator(dim, is_real, seed)
    assert_allclose(matrix, expected_mat)
