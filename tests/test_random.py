"""Tests for random."""
import numpy as np

from toqito.random import random_density_matrix
from toqito.random import random_ginibre
from toqito.random import random_povm
from toqito.random import random_state_vector
from toqito.random import random_unitary

from toqito.matrix_props import is_density
from toqito.matrix_props import is_unitary
from toqito.state_props import is_pure


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


def test_random_ginibre_dims():
    """Generate random Ginibre matrix and check proper dimensions."""
    gin_mat = random_ginibre(2, 2)
    np.testing.assert_equal(gin_mat.shape[0], 2)
    np.testing.assert_equal(gin_mat.shape[1], 2)


def test_random_povm_unitary_not_real():
    """Generate random POVMs and check that they sum to the identity."""
    dim, num_inputs, num_outputs = 2, 2, 2
    povms = random_povm(dim, num_inputs, num_outputs)
    np.testing.assert_equal(
        np.allclose(povms[:, :, 0, 0] + povms[:, :, 0, 1], np.identity(dim)), True
    )


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
