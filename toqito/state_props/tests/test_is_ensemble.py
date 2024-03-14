"""Test is_ensemble."""

import numpy as np
import pytest

from toqito.state_props import is_ensemble


@pytest.mark.parametrize(
    "states, expected_result",
    [
        # Test if valid ensemble returns True.
        ([np.array([[0.5, 0], [0, 0]]), np.array([[0, 0], [0, 0.5]])], True),
        # Test if non-valid (non-PSD) ensemble returns False.
        ([np.array([[0.5, 0], [0, 0]]), np.array([[-1, -1], [-1, -1]])], False),
    ],
)
def test_is_ensemble(states, expected_result):
    """Test function works as expected for a valid input."""
    np.testing.assert_equal(is_ensemble(states), expected_result)
