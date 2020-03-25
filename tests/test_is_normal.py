"""Tests for is_normal function."""
import unittest
import numpy as np

from toqito.matrix.properties.is_normal import is_normal


class TestIsNormal(unittest.TestCase):
    """Unit test for is_normal."""

    def test_is_normal(self):
        """Test that normal matrix returns True."""
        mat = np.identity(4)
        self.assertEqual(is_normal(mat), True)

    def test_is_not_normal(self):
        """Test that non-normal matrix returns False."""
        mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertEqual(is_normal(mat), False)


if __name__ == "__main__":
    unittest.main()
