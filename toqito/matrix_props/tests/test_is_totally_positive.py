"""Test is_totally_positive."""

import numpy as np
import pytest

from toqito.matrix_props import is_totally_positive


@pytest.mark.parametrize(
    "mat, tol, sub_sizes, expected_result",
    [
        # 2x2 matrix that is known to be totally positive.
        (np.array([[1, 2], [2, 5]]), 1e-6, None, True),
        # 2x2 matrix that is not totally positive.
        (np.array([[1, -2], [3, 4]]), 1e-6, None, False),
        # 3x3 that is totally positive.
        (np.array([[1, 2, 3], [2, 5, 8], [3, 8, 14]]), 1e-6, None, True),
        # 3x3 that is not totally positive
        (np.array([[1, 2, 3], [-1, -2, -3], [2, 5, 8]]), 1e-6, None, False),
        # Matrix with complex entries (which should not be totally positive).
        (np.array([[1 + 1j, 2], [3, 4]]), 1e-6, None, False),
        # Matrix that is borderline not totally positive due to a very small negative determinant.
        (np.array([[1, 2], [3, 4 - 1e-10]]), 1e-6, None, False),
        # Test specifying custom sizes for submatrices.
        (np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), 1e-6, [1, 2], False),
        # Test specifying a custom tolerance level.
        (np.array([[1, 2], [3, 4 - 1e-10]]), 1e-9, None, False),
        # Test a matrix that is just a single row or column.
        (np.array([[1, 2, 3]]), 1e-6, None, True),
        # Test the identity matrix, which should be totally positive.
        (np.identity(3), 1e-6, None, True),
    ],
)
def test_is_totally_positive(mat, tol, sub_sizes, expected_result):
    """Test function works as expected for a valid input."""
    np.testing.assert_equal(is_totally_positive(mat, tol, sub_sizes), expected_result)


@pytest.mark.parametrize(
    "mat, tol, sub_sizes",
    [
        # Empty matrix is an invalid input
        (np.array([]), 1e-6, None),
    ],
)
def test_is_totally_positive_invalid(mat, tol, sub_sizes):
    """Test function works as expected for an invalid input."""
    with np.testing.assert_raises(ValueError):
        is_totally_positive(mat, tol, sub_sizes)
