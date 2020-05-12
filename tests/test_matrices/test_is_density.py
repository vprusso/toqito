"""Tests for is_density function."""
import unittest
import numpy as np

from toqito.matrix_props import is_density
from toqito.random import random_density_matrix


class TestIsDensity(unittest.TestCase):
    """Unit test for is_density."""

    def test_is_density_ndarray(self):
        """Test if ndarray type is density matrix."""
        mat = random_density_matrix(2, True)
        self.assertEqual(is_density(mat), True)

    def test_is_density_npmatrix(self):
        """Test if np.matrix type is density matrix."""
        mat = np.matrix(random_density_matrix(4))
        self.assertEqual(is_density(mat), True)


if __name__ == "__main__":
    unittest.main()
