"""Tests for is_hermitian function."""
import unittest
import numpy as np

from toqito.matrix_props import is_hermitian


class TestIsHermitian(unittest.TestCase):

    """Unit test for is_hermitian."""

    def test_is_hermitian(self):
        """Test if matrix is Hermitian."""
        mat = np.array([[2, 2 + 1j, 4], [2 - 1j, 3, 1j], [4, -1j, 1]])
        self.assertEqual(is_hermitian(mat), True)

    def test_is_non_hermitian(self):
        """Test non-Hermitian matrix."""
        mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertEqual(is_hermitian(mat), False)


if __name__ == "__main__":
    unittest.main()
