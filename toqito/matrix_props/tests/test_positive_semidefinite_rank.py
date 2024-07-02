"""Tests for the PSD rank of a matrix."""

import numpy as np
import pytest

from toqito.matrix_props import positive_semidefinite_rank


@pytest.mark.parametrize(
    "mat, expected_psd_rank",
    [
        # The PSD rank of the identity matrix is the dimension of the matrix.
        (np.identity(3), 3),
        # The PSD rank of this matrix is known to be 2 :cite:`Heinosaari_2024_Can` (Equation 21).
        (1/2 * np.array([[0, 1, 1], [1,0,1], [1,1,0]]), 2),
    ],
)
def test_positive_semidefinite_rank(mat, expected_psd_rank):
    """Checks the PSD rank of known cases."""
    np.testing.assert_equal(positive_semidefinite_rank(mat), expected_psd_rank)

