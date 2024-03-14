"""Test unvec."""

import numpy as np
import pytest

from toqito.matrix_ops import unvec


@pytest.mark.parametrize(
    "vector, shape, expected_result",
    [
        # Test standard unvec operation on a vector.
        (np.array([1, 3, 2, 4]), None, np.array([[1, 2], [3, 4]])),
        # Test standard unvec operation on a vector with custom dimension.
        (np.array([1, 3, 2, 4]), [4, 1], np.array([[1], [3], [2], [4]])),
    ],
)
def test_unvec(vector, shape, expected_result):
    """Test function works as expected for a valid input."""
    np.testing.assert_array_equal(unvec(vector, shape), expected_result)
