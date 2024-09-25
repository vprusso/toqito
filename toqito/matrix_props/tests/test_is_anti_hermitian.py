"""Test is_anti_hermitian."""

import numpy as np
import pytest

from toqito.matrix_props import is_anti_hermitian

data = [
    # Test with anti-Hermitian matrix
    (np.array([[2j, -1 + 2j, 4j], [1 + 2j, 3j, -1], [4j, 1, 1j]]), True),
    # Test with non-anti-Hermitian matrix
    (np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), False),
    # Test with non-square matrix
    (np.array([[-1, 1, 1], [1, 2, 3]]), False),
]

@pytest.mark.parametrize("mat,expected_bool", data)
def test_is_anti_hermitian(mat, expected_bool):
    """Test if matrix is anti-Hermitian."""
    np.testing.assert_equal(is_anti_hermitian(mat), expected_bool)
