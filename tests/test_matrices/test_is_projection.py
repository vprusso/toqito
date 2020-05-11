"""Tests for is_projection function."""
import unittest
import numpy as np

from toqito.matrix_props import is_projection


class TestIsProjection(unittest.TestCase):

    """Unit test for is_projection."""

    def test_is_projection(self):
        """Check that projection matrix returns True."""
        mat = np.array([[0, 1], [0, 1]])
        self.assertEqual(is_projection(mat), True)

    def test_is_projection_2(self):
        """Check that projection matrix returns True."""
        mat = np.array([[1, 0], [0, 1]])
        self.assertEqual(is_projection(mat), True)

    def test_is_not_pd(self):
        """Check that non-projection matrix returns False."""
        mat = np.array([[-1, -1], [-1, -1]])
        self.assertEqual(is_projection(mat), False)

    def test_is_not_pd2(self):
        """Check that non-projection matrix returns False."""
        mat = np.array([[1, 2, 3], [2, 1, 4]])
        self.assertEqual(is_projection(mat), False)


if __name__ == "__main__":
    unittest.main()
