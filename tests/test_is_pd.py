"""Tests for is_pd function."""
import unittest
import numpy as np

from toqito.linear_algebra.properties.is_pd import is_pd


class TestIsPD(unittest.TestCase):
    """Unit test for is_pd."""

    def test_is_pd(self):
        """Check that positive definite matrix returns True."""
        mat = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
        self.assertEqual(is_pd(mat), True)

    def test_is_not_pd(self):
        """Check that non-positive definite matrix returns False."""
        mat = np.array([[-1, -1], [-1, -1]])
        self.assertEqual(is_pd(mat), False)

    def test_is_not_pd2(self):
        """Check that non-square matrix returns False."""
        mat = np.array([[1, 2, 3], [2, 1, 4]])
        self.assertEqual(is_pd(mat), False)


if __name__ == "__main__":
    unittest.main()
