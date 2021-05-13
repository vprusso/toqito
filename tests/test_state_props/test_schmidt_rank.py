"""Test schmidt_rank."""
import numpy as np

from toqito.state_props import schmidt_rank
from toqito.states import basis
from toqito.states import bell


def test_schmidt_rank_bell_state():
    """
    Computing the Schmidt rank of the entangled Bell state should yield a
    value greater than 1.
    """
    np.testing.assert_equal(np.isclose(schmidt_rank(bell(0)), 2), True)


def test_schmidt_rank_bell_state_dim_1():
    """
    Computing the Schmidt rank of Bell state with dim 1.
    """
    np.testing.assert_equal(np.isclose(schmidt_rank(bell(0), 1), 1), True)


def test_schmidt_rank_bell_state_list():
    """
    Computing the Schmidt rank of Bell state with list as argument for dims.
    """
    np.testing.assert_equal(np.isclose(schmidt_rank(bell(0), [2, 2]), 2), True)


def test_schmidt_rank_singlet_state():
    """
    Computing the Schmidt rank of the entangled singlet state should yield
    a value greater than 1.
    """
    e_0, e_1 = basis(2, 0), basis(2, 1)
    rho = 1 / np.sqrt(2) * (np.kron(e_0, e_1) - np.kron(e_1, e_0))
    np.testing.assert_equal(schmidt_rank(rho) > 1, True)


def test_schmidt_rank_separable_state():
    """
    Computing the Schmidt rank of a separable state should yield a value
    equal to 1.
    """
    e_0, e_1 = basis(2, 0), basis(2, 1)
    e_00 = np.kron(e_0, e_0)
    e_01 = np.kron(e_0, e_1)
    e_10 = np.kron(e_1, e_0)
    e_11 = np.kron(e_1, e_1)
    rho = 1 / 2 * (e_00 - e_01 - e_10 + e_11)
    np.testing.assert_equal(schmidt_rank(rho) == 1, True)


def test_schmidt_rank_entangled_state():
    """Computing Schmidt rank of entangled state should be > 1."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    phi = (
        (1 + np.sqrt(6)) / (2 * np.sqrt(6)) * np.kron(e_0, e_0)
        + (1 - np.sqrt(6)) / (2 * np.sqrt(6)) * np.kron(e_0, e_1)
        + (np.sqrt(2) - np.sqrt(3)) / (2 * np.sqrt(6)) * np.kron(e_1, e_0)
        + (np.sqrt(2) + np.sqrt(3)) / (2 * np.sqrt(6)) * np.kron(e_1, e_1)
    )
    np.testing.assert_equal(schmidt_rank(phi) == 2, True)


def test_schmidt_rank_separable_density_matrix():
    """Computing Schmidt rank of separable density matrix should be 1."""
    np.testing.assert_equal(schmidt_rank(np.identity(4)) == 1, True)
    np.testing.assert_equal(schmidt_rank(np.identity(16)) == 1, True)


def test_schmidt_rank_bell_density_matrices():
    """Computing Schmidt rank of Bell density matrices should be 4."""
    np.testing.assert_equal(schmidt_rank(bell(0) * bell(0).conj().T) == 4, True)
    np.testing.assert_equal(schmidt_rank(bell(1) * bell(1).conj().T) == 4, True)
    np.testing.assert_equal(schmidt_rank(bell(2) * bell(2).conj().T) == 4, True)
    np.testing.assert_equal(schmidt_rank(bell(3) * bell(3).conj().T) == 4, True)

    np.testing.assert_equal(schmidt_rank(bell(0) * bell(0).conj().T, [2, 2]) == 4, True)
    np.testing.assert_equal(schmidt_rank(bell(0) * bell(0).conj().T, 2) == 4, True)


if __name__ == "__main__":
    np.testing.run_module_suite()
