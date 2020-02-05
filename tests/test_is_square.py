"""Tests for is_square function."""
import unittest
import numpy as np

from toqito.matrix.properties.is_square import is_square


class TestIsSquare(unittest.TestCase):
    """Unit test for is_square."""

    def test_is_square(self):
        """Test that square matrix returns True."""
        mat = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]])
        self.assertEqual(is_square(mat), True)

    def test_is_not_square(self):
        """Test that non-square matrix returns False."""
        mat = np.array([[1, 2, 3],
                        [4, 5, 6]])
        self.assertEqual(is_square(mat), False)


if __name__ == '__main__':
    unittest.main()
