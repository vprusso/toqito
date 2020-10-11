"""Tests for fidelity."""
import cvxpy
import numpy as np


from toqito.state_metrics import fidelity
from toqito.states import basis


def test_fidelity_default():
    """Test fidelity default arguments."""
    rho = np.array([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]])
    sigma = rho

    res = fidelity(rho, sigma)
    np.testing.assert_equal(np.isclose(res, 1), True)


def test_fidelity_cvx():
    """Test fidelity for cvx objects."""
    rho = cvxpy.bmat([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]])
    sigma = rho

    res = fidelity(rho, sigma)
    np.testing.assert_equal(np.isclose(res, 1), True)


def test_fidelity_non_identical_states_1():
    """Test the fidelity between two non-identical states."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    rho = 3 / 4 * e_0 * e_0.conj().T + 1 / 4 * e_1 * e_1.conj().T
    sigma = 2 / 3 * e_0 * e_0.conj().T + 1 / 3 * e_1 * e_1.conj().T
    np.testing.assert_equal(np.isclose(fidelity(rho, sigma), 0.996, rtol=1e-03), True)


def test_fidelity_non_identical_states_2():
    """Test the fidelity between two non-identical states."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    rho = 3 / 4 * e_0 * e_0.conj().T + 1 / 4 * e_1 * e_1.conj().T
    sigma = 1 / 8 * e_0 * e_0.conj().T + 7 / 8 * e_1 * e_1.conj().T
    np.testing.assert_equal(np.isclose(fidelity(rho, sigma), 0.774, rtol=1e-03), True)


def test_fidelity_non_square():
    """Tests for invalid dim."""
    rho = np.array([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]])
    sigma = np.array([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]])
    with np.testing.assert_raises(ValueError):
        fidelity(rho, sigma)


def test_fidelity_invalid_dim():
    """Tests for invalid dim."""
    rho = np.array([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]])
    sigma = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    with np.testing.assert_raises(ValueError):
        fidelity(rho, sigma)


if __name__ == "__main__":
    np.testing.run_module_suite()
