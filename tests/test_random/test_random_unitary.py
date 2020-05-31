"""Test random_unitary."""
import numpy as np

from toqito.random import random_unitary
from toqito.matrix_props import is_unitary


def test_random_unitary_not_real():
    """Generate random non-real unitary matrix."""
    mat = random_unitary(2)
    np.testing.assert_equal(is_unitary(mat), True)


def test_random_unitary_real():
    """Generate random real unitary matrix."""
    mat = random_unitary(2, True)
    np.testing.assert_equal(is_unitary(mat), True)


def test_random_unitary_vec_dim():
    """Generate random non-real unitary matrix."""
    mat = random_unitary([4, 4], True)
    np.testing.assert_equal(is_unitary(mat), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
