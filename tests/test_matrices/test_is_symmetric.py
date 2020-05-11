"""Tests for is_symmetric function."""
import unittest
import numpy as np

from toqito.matrix_props import is_symmetric


class TestIsSymmetric(unittest.TestCase):
    
    """Unit test for is_symmetric."""

    def test_is_symmetric(self):
        """Test that symmetric matrix returns True."""
        mat = np.array([[1, 7, 3], [7, 4, -5], [3, -5, 6]])
        self.assertEqual(is_symmetric(mat), True)

    def test_is_not_symmetric(self):
        """Test that non-symmetric matrix returns False."""
        mat = np.array([[1, 2], [3, 4]])
        self.assertEqual(is_symmetric(mat), False)


if __name__ == "__main__":
    unittest.main()
