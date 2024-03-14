"""Test is_positive_definite."""

import numpy as np

from toqito.matrix_props import is_positive_definite


def test_is_is_positive_definite():
    """Check that positive definite matrix returns True."""
    mat = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
    np.testing.assert_equal(is_positive_definite(mat), True)


def test_is_not_positive_definite():
    """Check that non-positive definite matrix returns False."""
    mat = np.array([[-1, -1], [-1, -1]])
    np.testing.assert_equal(is_positive_definite(mat), False)

    eps = 0
    c_var = 1 / np.sqrt(3)
    gram = np.array(
        [
            [1, c_var, c_var, c_var],
            [c_var, 1, c_var * 1j, (1 + c_var * 1j) / 2],
            [c_var, -c_var * 1j, 1, (1 - c_var * 1j) / 2],
            [c_var, (1 - c_var * 1j) / 2, (1 + c_var * 1j) / 2, 1],
        ]
    )

    v_vec = np.array(
        [
            [1],
            [(-np.sqrt(3) + 1j) / 2],
            [(-np.sqrt(3) - 1j) / 2],
            [0],
        ]
    )
    w_vec = np.array(
        [
            [0],
            [0],
            [0],
            [1],
        ]
    )
    gram_eps = 1 / (1 - 2 * eps) * (gram + eps * (v_vec @ v_vec.conj().T + w_vec @ w_vec.conj().T - 3 * np.identity(4)))
    np.testing.assert_equal(is_positive_definite(gram_eps), False)


def test_is_positive_definite_not_hermitian():
    """Input must be a Hermitian matrix."""
    mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    np.testing.assert_equal(is_positive_definite(mat), False)
