"""Tests for random_density_matrix function."""
import unittest

from toqito.random.random_density_matrix import random_density_matrix
from toqito.matrix.properties.is_density import is_density


class TestRandomDensity(unittest.TestCase):
    """Unit test for random_unitary."""

    def test_random_density_not_real(self):
        """Generate random non-real density matrix."""
        mat = random_density_matrix(2)
        self.assertEqual(is_density(mat), True)

    def test_random_density_real(self):
        """Generate random real density matrix."""
        mat = random_density_matrix(2, True)
        self.assertEqual(is_density(mat), True)

    def test_random_density_not_real_bures(self):
        """Generate random non-real density matrix according to Bures metric."""
        mat = random_density_matrix(2, distance_metric="bures")
        self.assertEqual(is_density(mat), True)

    def test_random_density_not_real_k_param(self):
        """Generate random non-real density matrix wih k_param."""
        mat = random_density_matrix(2, distance_metric="bures")
        self.assertEqual(is_density(mat), True)

    def test_random_density_not_real_all_params(self):
        """Generate random non-real density matrix all params."""
        mat = random_density_matrix(2, True, 2, "haar")
        self.assertEqual(is_density(mat), True)


if __name__ == '__main__':
    unittest.main()
