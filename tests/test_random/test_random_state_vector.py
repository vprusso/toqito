"""Test random_state_vector."""
import numpy as np

from toqito.random import random_state_vector
from toqito.state_props import is_pure


def test_random_state_vector_complex_state_purity():
    """Check that complex state vector from random state vector is pure."""
    vec = random_state_vector(2)
    mat = vec.conj().T * vec
    np.testing.assert_equal(is_pure(mat), True)


def test_random_state_vector_complex_state_purity_k_param():
    """Check that complex state vector with k_param > 0."""
    vec = random_state_vector(2, False, 1)
    mat = vec.conj().T * vec
    np.testing.assert_equal(is_pure(mat), False)


def test_random_state_vector_complex_state_purity_k_param_dim_list():
    """Check that complex state vector with k_param > 0 and dim list."""
    vec = random_state_vector([2, 2], False, 1)
    mat = vec.conj().T * vec
    np.testing.assert_equal(is_pure(mat), False)


def test_random_state_vector_real_state_purity_with_k_param():
    """Check that real state vector with k_param > 0."""
    vec = random_state_vector(2, True, 1)
    mat = vec.conj().T * vec
    np.testing.assert_equal(is_pure(mat), False)


def test_random_state_vector_real_state_purity():
    """Check that real state vector from random state vector is pure."""
    vec = random_state_vector(2, True)
    mat = vec.conj().T * vec
    np.testing.assert_equal(is_pure(mat), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
