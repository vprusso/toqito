"""Test update_odometer."""

import numpy as np
import pytest

from toqito.helper import update_odometer


@pytest.mark.parametrize(
    "test_vec, upper_lim, expected",
    [
        # Update odometer from [2, 2] to [0, 0].
        (np.array([2, 2]), np.array([3, 2]), [0, 0]),
        # Update odometer from [0, 0] to [0, 1].
        (np.array([0, 0]), np.array([3, 2]), [0, 1]),
        # Update odometer from [0, 1] to [1, 0].
        (np.array([0, 1]), np.array([3, 2]), [1, 0]),
        # Update odometer from [1, 1] to [2, 0].
        (np.array([1, 1]), np.array([3, 2]), [2, 0]),
        # Update odometer from [2, 0] to [2, 1].
        (np.array([2, 0]), np.array([3, 2]), [2, 1]),
        # Update odometer from [2, 1] to [0, 0].
        (np.array([2, 1]), np.array([3, 2]), [0, 0]),
        # Return `None` if empty lists are provided.
        (np.array([]), np.array([]), []),
    ],
)
def test_update_odometer(test_vec, upper_lim, expected):
    """Test function works correctly."""
    assert (update_odometer(test_vec, upper_lim) == expected).all()
