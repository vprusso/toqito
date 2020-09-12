"""Test in_separable_ball."""
import numpy as np

from toqito.random import random_unitary
from toqito.state_props import in_separable_ball


def test_in_separable_ball_matrix_true():
    """Test matrix in separable ball returns True."""
    u_mat = random_unitary(4)
    lam = np.array([1, 1, 1, 0]) / 3
    rho = u_mat @ np.diag(lam) @ u_mat.conj().T
    np.testing.assert_equal(in_separable_ball(rho), True)


def test_in_separable_ball_matrix_false():
    """Test matrix not in separable ball returns False."""
    u_mat = random_unitary(4)
    lam = np.array([1.01, 1, 0.99, 0]) / 3
    rho = u_mat @ np.diag(lam) @ u_mat.conj().T
    np.testing.assert_equal(in_separable_ball(rho), False)


def test_in_separable_ball_trace_lt_dim():
    """Test for case when trace of matrix is less than the largest dim."""
    rho = np.zeros((4, 4))
    np.testing.assert_equal(in_separable_ball(rho), False)


def test_in_separable_ball_eigs_false():
    """Test eigenvalues of matrix not in separable ball returns False."""
    u_mat = random_unitary(4)
    lam = np.array([1.01, 1, 0.99, 0]) / 3
    rho = u_mat @ np.diag(lam) @ u_mat.conj().T

    eig_vals = np.linalg.eigvalsh(rho)
    np.testing.assert_equal(in_separable_ball(eig_vals), False)


if __name__ == "__main__":
    np.testing.run_module_suite()
