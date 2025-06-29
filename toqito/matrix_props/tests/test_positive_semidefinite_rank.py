"""Tests for the PSD rank of a matrix."""

import numpy as np
import pytest

from toqito.matrix_props import positive_semidefinite_rank


@pytest.mark.parametrize(
    "mat, max_rank, expected_psd_rank",
    [
        # The PSD rank of the identity matrix is the dimension of the matrix.
        (np.identity(3), 10, 3),
        # If the max_rank is lower than the actual rank, the function returns None.
        (np.identity(3), 2, None),
        # The PSD rank of this matrix is known to be 2 :footcite:`Heinosaari_2024_Can` (Equation 21).
        (1 / 2 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]), 10, 2),
    ],
)
def test_positive_semidefinite_rank(mat, max_rank, expected_psd_rank):
    """Checks the PSD rank of known cases."""
    np.testing.assert_equal(positive_semidefinite_rank(mat, max_rank), expected_psd_rank)


@pytest.mark.parametrize(
    "mat, expected_msg",
    [
        # Cannot compute PSD rank of negative matrix.
        (
            np.array([[-1, 2, 3], [4, -5, 6], [7, 8, 9]]),
            "Matrix must be nonnegative.",
        ),
        # Cannot compute PSD rank of non-square matrix.
        (
            np.array([[0, 1, 1], [1, 0, 1]]),
            "Matrix must be square.",
        ),
    ],
)
def test_positive_semidefinite_rank_raises_error(mat, expected_msg):
    """Ensure PSD rank catches non-compliant input matrices."""
    with pytest.raises(ValueError, match=expected_msg):
        positive_semidefinite_rank(mat)
