"""Tests for is_diagonal function."""
import unittest
import numpy as np

from toqito.linear_algebra.properties.is_diagonal import is_diagonal


class TestIsDiagonal(unittest.TestCase):
    """Unit test for is_diagonal."""

    def test_is_diagonal(self):
        """Test if matrix is diagonal."""
        mat = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        self.assertEqual(is_diagonal(mat), True)

    def test_is_non_diagonal(self):
        """Test non-diagonal matrix."""
        mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertEqual(is_diagonal(mat), False)

    def test_non_square(self):
        """Test on a non-square matrix."""
        mat = np.array([[1, 0, 0], [0, 1, 0]])
        self.assertEqual(is_diagonal(mat), False)


if __name__ == "__main__":
    unittest.main()
