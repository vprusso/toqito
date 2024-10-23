"""Test random_state_vector."""
import numpy as np
import pytest
from numpy.ma.testutils import assert_array_almost_equal

from toqito.rand import random_state_vector
from toqito.state_props import is_pure


@pytest.mark.parametrize(
    "dim, is_real, k_param",
    [
        # Check that complex state vector from random state vector is pure.
        (2, False, 0),
        # Check that complex state vector with k_param > 0.
        (2, False, 1),
        # Check that complex state vector with k_param > 0 and dim list.
        ([2, 2], False, 1),
        # Check that real state vector with k_param > 0.
        (2, True, 1),
        # Check that real state vector from random state vector is pure.
        (2, True, 0),
    ],
)
def test_random_state_vector(dim, is_real, k_param):
    """Test function works as expected for a valid input."""
    # We expect the density matrix of any random state vector to be pure.
    vec = random_state_vector(dim=dim, is_real=is_real, k_param=k_param).reshape(-1, 1)
    mat = vec @ vec.conj().T
    assert is_pure(mat)

@pytest.mark.parametrize(
    "dim, is_real, k_param, expected",
    [
        (
            2,
            False,
            0,
            np.array([
                [0.91920422+0.29684938j],
                [0.07250293+0.24836943j]
            ])
        ),
        (
            2,
            False,
            1,
            np.array([
                -0.01113702 + 0.61768143j,
                0.07125721 + 0.20424701j,
                -0.68156797 + 0.21116929j,
                -0.19827006 + 0.15202896j
            ])
        ),
        (
            [2, 2],
            False,
            1,
            np.array([
                -0.01113702 + 0.61768143j,
                0.07125721 + 0.20424701j,
                -0.68156797 + 0.21116929j,
                -0.19827006 + 0.15202896j
            ])
        ),
        (
            2,
            True,
            1,
            np.array([0.76458086, 0.63971337, 0.06030689, 0.05045788])
        ),
        (
            2,
            True,
            0,
            np.array([
                [0.99690375],
                [0.07863154]
            ])
        ),
    ],
)
def test_seed(dim, is_real, k_param, expected):
    """Test that the function returns the expected output when seeded."""
    vec = random_state_vector(dim, is_real=is_real, k_param=k_param, seed=123)
    assert_array_almost_equal(vec, expected)
