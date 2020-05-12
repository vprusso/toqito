"""Tests for is_density function."""
import unittest

from toqito.matrix_props import is_density
from toqito.random import random_density_matrix


class TestIsDensity(unittest.TestCase):
    """Unit test for is_density."""

    def test_is_density_real_entries(self):
        """Test if random density matrix with real entries is density matrix."""
        mat = random_density_matrix(2, True)
        self.assertEqual(is_density(mat), True)

    def test_is_density_complex_entries(self):
        """Test if density matrix with complex entries is density matrix."""
        mat = random_density_matrix(4)
        self.assertEqual(is_density(mat), True)


if __name__ == "__main__":
    unittest.main()
