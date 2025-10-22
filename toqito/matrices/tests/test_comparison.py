"""Test comparison."""

import numpy as np
import pytest

from toqito.matrices.comparison import comparison


@pytest.mark.parametrize(
    "mat, expected",
    [
        # 1x1 matrix.
        (np.array([[-3]]), np.array([[3]])),
        # 2x2 matrix.
        (np.array([[2, -1], [3, 4]]), np.array([[2, -1], [-3, 4]])),
        # 2x2 matrix with complex entries.
        (
            np.array([[1 + 2j, -3j], [4 - 5j, -6 + 7j]]),
            np.array([[abs(1 + 2j), -abs(-3j)], [-abs(4 - 5j), abs(-6 + 7j)]]),
        ),
        # 3x3 matrix.
        (
            np.array([[0, -1, 2], [3, 4, -5], [-6, 7, 8]]),
            np.array([[abs(0), -abs(-1), -abs(2)], [-abs(3), abs(4), -abs(-5)], [-abs(-6), -abs(7), abs(8)]]),
        ),
        # All-zero matrix.
        (np.zeros((2, 2)), np.zeros((2, 2))),
    ],
)
def test_comparison_matrix(mat, expected):
    """Test that the correct comparison matrix is generated."""
    np.testing.assert_allclose(comparison(mat), expected)


def test_non_square_matrix():
    """Ensure non-square matrices are flagged."""
    mat = np.array([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError, match="Input matrix must be square."):
        comparison(mat)
