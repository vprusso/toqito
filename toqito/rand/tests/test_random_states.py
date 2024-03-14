"""Test random_states."""

import numpy as np
import pytest

from toqito.rand import random_states
from toqito.state_props import is_pure


@pytest.mark.parametrize(
    "num_states, dim",
    [
        # Test with a single quantum state of dimension 2.
        (1, 2),
        # Test with multiple quantum states of the same dimension.
        (3, 2),
        # Test with a single quantum state of higher dimension.
        (1, 4),
        # Test with multiple quantum states of higher dimension.
        (2, 4),
    ],
)
def test_random_states(num_states, dim):
    """Test for random_states function."""
    # Generate a list of random quantum states.
    states = random_states(num_states, dim)

    # Ensure the number of states generated is as expected.
    assert len(states) == num_states

    # Check each state is a valid quantum state.
    for state in states:
        assert state.shape == (dim, 1)
        # Convert state vector to density matrix.
        dm = np.outer(state, np.conj(state))
        # Verify each state is pure.
        assert is_pure(dm)
