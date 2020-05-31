"""Test random_ginibre."""
import numpy as np

from toqito.random import random_ginibre


def test_random_ginibre_dims():
    """Generate random Ginibre matrix and check proper dimensions."""
    gin_mat = random_ginibre(2, 2)
    np.testing.assert_equal(gin_mat.shape[0], 2)
    np.testing.assert_equal(gin_mat.shape[1], 2)


if __name__ == "__main__":
    np.testing.run_module_suite()
