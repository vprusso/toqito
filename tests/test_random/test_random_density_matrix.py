"""Test random_density_matrix."""
import numpy as np

from toqito.random import random_density_matrix
from toqito.matrix_props import is_density


def test_random_density_not_real():
    """Generate random non-real density matrix."""
    mat = random_density_matrix(2)
    np.testing.assert_equal(is_density(mat), True)


def test_random_density_real():
    """Generate random real density matrix."""
    mat = random_density_matrix(2, True)
    np.testing.assert_equal(is_density(mat), True)


def test_random_density_not_real_bures():
    """Random non-real density matrix according to Bures metric."""
    mat = random_density_matrix(2, distance_metric="bures")
    np.testing.assert_equal(is_density(mat), True)


def test_random_density_not_real_k_param():
    """Generate random non-real density matrix wih k_param."""
    mat = random_density_matrix(2, distance_metric="bures")
    np.testing.assert_equal(is_density(mat), True)


def test_random_density_not_real_all_params():
    """Generate random non-real density matrix all params."""
    mat = random_density_matrix(2, True, 2, "haar")
    np.testing.assert_equal(is_density(mat), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
