"""Test is_mutually_orthogonal."""
import numpy as np
import pytest

from toqito.state_props import is_mutually_orthogonal
from toqito.states import bell


@pytest.mark.parametrize("states, expected_result", [
    # Return True for orthogonal Bell vectors.
    ([bell(0), bell(1), bell(2), bell(3)], True),
    # Return False for non-orthogonal vectors.
    ([np.array([1, 0]), np.array([1, 1])], False),
])
def test_is_mutually_orthogonal(states, expected_result):
    np.testing.assert_equal(is_mutually_orthogonal(states), expected_result)


@pytest.mark.parametrize("states", [
    # Tests for invalid input len.
    ([np.array([1, 0])]),
])
def test_is_mutually_orthogonal_basis_invalid_input(states):
    with np.testing.assert_raises(ValueError):
        is_mutually_orthogonal(states)
