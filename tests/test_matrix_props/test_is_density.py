"""Test is_density."""
import numpy as np

from toqito.matrix_props import is_density
from toqito.random import random_density_matrix


def test_is_density_real_entries():
    """Test if random density matrix with real entries is density matrix."""
    mat = random_density_matrix(2, True)
    np.testing.assert_equal(is_density(mat), True)


def test_is_density_complex_entries():
    """Test if density matrix with complex entries is density matrix."""
    mat = random_density_matrix(4)
    np.testing.assert_equal(is_density(mat), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
