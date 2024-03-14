"""Test random_state_vector."""

import pytest

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
