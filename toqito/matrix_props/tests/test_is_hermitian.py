"""Test is_hermitian."""

import numpy as np
import pytest

from toqito.matrix_props import is_hermitian

data = [
    # Test with Hermitian matrix
    (np.array([[2, 2 + 1j, 4], [2 - 1j, 3, 1j], [4, -1j, 1]]), True),
    # Test with non-Hermitian matrix
    (np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), False),
    # Test with non-square matrix
    (np.array([[-1, 1, 1], [1, 2, 3]]), False),
]

@pytest.mark.parametrize("mat,expected_bool", data)
def test_is_hermitian(mat, expected_bool):
    """Test if matrix is Hermitian."""
    np.testing.assert_equal(is_hermitian(mat), expected_bool)
