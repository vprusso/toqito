"""Test trine."""

import numpy as np

from toqito.states import trine


def test_trine():
    """Test function works as expected for a valid input."""
    states = trine()

    # Trine[0]
    np.testing.assert_array_equal(
        states[0],
        np.array([[1], [0]]),
    )

    # Trine[1]
    np.testing.assert_array_equal(
        states[1],
        -1 / 2 * (np.array([[1], [0]]) + np.sqrt(3) * np.array([[0], [1]])),
    )

    # Trine[2]
    np.testing.assert_array_equal(
        states[2],
        -1 / 2 * (np.array([[1], [0]]) - np.sqrt(3) * np.array([[0], [1]])),
    )
