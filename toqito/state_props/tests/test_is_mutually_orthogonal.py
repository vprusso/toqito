"""Test is_mutually_orthogonal."""

import numpy as np
import pytest

from toqito.state_props import is_mutually_orthogonal
from toqito.states import bell


@pytest.mark.parametrize(
    "states, expected_result",
    [
        # Return True for orthogonal Bell vectors.
        ([bell(0), bell(1), bell(2), bell(3)], True),
        # Return False for non-orthogonal vectors.
        ([np.array([1, 0]), np.array([1, 1])], False),
        # Orthogonal vectors in R^2
        ([np.array([1, 0]), np.array([0, 1])], True),
        # Orthogonal vectors in R^2
        ([np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])], True),
        # Orthogonal complex-valued vectors
        ([np.array([[1], [1j]]), np.array([[1j], [1]])], True),
        # Vectors with zero elements.
        ([np.array([[0], [0]]), np.array([[1], [0]])], True),
        # Colinear vectors.
        ([np.array([[1], [2]]), np.array([2, 4])], False),
        # Vectors that are theoretically orthogonal but due to numerical precision issues might not
        # be exactly orthogonal.
        ([np.array([[1], [np.sqrt(2)]]), np.array([[-np.sqrt(2)], [1]])], True),
    ],
)
def test_is_mutually_orthogonal(states, expected_result):
    """Test function works as expected for a valid input."""
    np.testing.assert_equal(is_mutually_orthogonal(states), expected_result)


@pytest.mark.parametrize(
    "states",
    [
        # Tests for invalid input len.
        ([np.array([1, 0])]),
        # Single vector should raise error.
        ([np.array([[1], [2], [3]])]),
        # Vectors of differing lengths.
        ([np.array([[1], [0]]), np.array([[1], [0], [1]])]),
        # Empty vector.
        ([]),
    ],
)
def test_is_mutually_orthogonal_basis_invalid_input(states):
    """Test function works as expected for an invalid input."""
    with np.testing.assert_raises(ValueError):
        is_mutually_orthogonal(states)
