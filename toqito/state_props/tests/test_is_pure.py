"""Test is_pure."""

import numpy as np
import pytest

from toqito.state_props import is_pure
from toqito.states import bell

e_0, e_1, e_2 = np.array([[1], [0], [0]]), np.array([[0], [1], [0]]), np.array([[0], [0], [1]])


@pytest.mark.parametrize(
    "state, expected_result",
    [
        # Ensure that pure Bell state returns True.
        (bell(0) @ bell(0).conj().T, True),
        # Check that list of pure states returns True.
        ([e_0 @ e_0.conj().T, e_1 @ e_1.conj().T, e_2 @ e_2.conj().T], True),
        # Check that non-pure state returns False.
        (np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), False),
        # Check that list of non-pure states return False.
        (
            [
                np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                np.array([[1, 2, 3], [10, 11, 12], [7, 8, 9]]),
            ],
            False,
        ),
    ],
)
def test_is_pure_state(state, expected_result):
    """Test function works as expected for a valid input."""
    np.testing.assert_equal(is_pure(state), expected_result)
