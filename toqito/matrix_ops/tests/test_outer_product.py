"""Test outer_product."""

import numpy as np
import pytest

from toqito.matrix_ops import outer_product


@pytest.mark.parametrize(
    "v1, v2, expected_result",
    [
        # Test with two vectors, no complications.
        (np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([[4, 5, 6], [8, 10, 12], [12, 15, 18]])),
        # Test with two vectors, with negative input/output values.
        (np.array([-1, 2, 3]), np.array([4, 5, 6]), np.array([[-4, -5, -6], [8, 10, 12], [12, 15, 18]])),
    ],
)
def test_outer_product(v1, v2, expected_result):
    """Test function works as expected for a valid input."""
    np.testing.assert_array_equal(outer_product(v1, v2), expected_result)


@pytest.mark.parametrize(
    "v1, v2",
    [
        # Different dimensions of vectors.
        (np.array([1, 2, 3]), np.array([4, 5, 6, 7])),
        # Vector and 2D array.
        (np.array([1, 2, 3]), np.array([[4, 5, 6], [7, 8, 9]])),
    ],
)
def test_outer_product_invalid_input(v1, v2):
    """Test function works as expected for an invalid input."""
    with pytest.raises(ValueError):
        outer_product(v1, v2)
