"""Tests for matsumoto_fidelity."""
import cvxpy
import numpy as np


from toqito.state_metrics import matsumoto_fidelity
from toqito.states import basis


def test_matsumoto_fidelity_default():
    """Test Matsumoto fidelity default arguments."""
    rho = np.array([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]])
    sigma = rho

    res = matsumoto_fidelity(rho, sigma)
    np.testing.assert_equal(np.isclose(res, 1), True)


def test_matsumoto_fidelity_cvx():
    """Test Matsumoto fidelity for cvx objects."""
    rho = cvxpy.bmat([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]])
    sigma = rho

    res = matsumoto_fidelity(rho, sigma)
    np.testing.assert_equal(np.isclose(res, 1), True)


def test_matsumoto_fidelity_non_identical_states_1():
    """Test the Matsumoto fidelity between two non-identical states."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    rho = 3 / 4 * e_0 * e_0.conj().T + 1 / 4 * e_1 * e_1.conj().T
    sigma = 2 / 3 * e_0 * e_0.conj().T + 1 / 3 * e_1 * e_1.conj().T
    np.testing.assert_equal(np.isclose(matsumoto_fidelity(rho, sigma), 0.996, rtol=1e-03), True)


def test_matsumoto_fidelity_non_identical_states_2():
    """Test the Matsumoto fidelity between two non-identical states."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    rho = 3 / 4 * e_0 * e_0.conj().T + 1 / 4 * e_1 * e_1.conj().T
    sigma = 1 / 8 * e_0 * e_0.conj().T + 7 / 8 * e_1 * e_1.conj().T
    np.testing.assert_equal(np.isclose(matsumoto_fidelity(rho, sigma), 0.774, rtol=1e-03), True)


def test_matsumoto_fidelity_non_square():
    """Tests for invalid dim."""
    rho = np.array([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]])
    sigma = np.array([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]])
    with np.testing.assert_raises(ValueError):
        matsumoto_fidelity(rho, sigma)


def test_matsumoto_fidelity_invalid_dim():
    """Tests for invalid dim."""
    rho = np.array([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]])
    sigma = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    with np.testing.assert_raises(ValueError):
        matsumoto_fidelity(rho, sigma)


if __name__ == "__main__":
    np.testing.run_module_suite()
