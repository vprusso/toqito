"""Test inner_product."""

import numpy as np
import pytest

from toqito.matrix_ops import inner_product


@pytest.mark.parametrize(
    "v1, v2, expected_result",
    [
        # Test with two vectors, no complications.
        (np.array([1, 2, 3]), np.array([4, 5, 6]), 32),
        # Test with two vectors, with negative input value.
        (np.array([-1, 2, 3]), np.array([4, 5, 6]), 24),
        # Test with two vectors, with negative expected output.
        (np.array([1, 2, -3]), np.array([4, 5, 6]), -4),
    ],
)
def test_inner_product(v1, v2, expected_result):
    """Test function works as expected for valid input."""
    assert inner_product(v1, v2) == expected_result


@pytest.mark.parametrize(
    "v1, v2",
    [
        # Different dimensions of vectors.
        (np.array([1, 2, 3]), np.array([4, 5, 6, 7])),
        # Vector and 2D array.
        (np.array([1, 2, 3]), np.array([[4, 5, 6], [7, 8, 9]])),
    ],
)
def test_inner_product_invalid_input(v1, v2):
    """Test function works as expected for an invalid input."""
    with pytest.raises(ValueError):
        inner_product(v1, v2)
