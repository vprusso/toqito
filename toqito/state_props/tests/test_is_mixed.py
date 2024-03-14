"""Test is_mixed."""

import numpy as np
import pytest

from toqito.state_props import is_mixed

e_0, e_1 = np.array([[1], [0]]), np.array([[0], [1]])


@pytest.mark.parametrize(
    "states, expected_result",
    [
        # Return True for mixed quantum state.
        (3 / 4 * e_0 @ e_0.conj().T + 1 / 4 * e_1 @ e_1.conj().T, True),
    ],
)
def test_is_mixed(states, expected_result):
    """Test function works as expected for a valid input."""
    np.testing.assert_equal(is_mixed(states), expected_result)
