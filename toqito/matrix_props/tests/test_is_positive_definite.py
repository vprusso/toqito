"""Test is_positive_definite."""

import numpy as np

from toqito.matrix_props import is_positive_definite


def test_is_positive_definite():
    """Check that positive definite matrix returns True."""
    mat = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
    np.testing.assert_equal(is_positive_definite(mat), True)


def test_is_not_positive_definite_negative_definite():
    """Check that negative definite matrix returns False."""
    mat = np.array([[-1, -1], [-1, -1]])
    np.testing.assert_equal(is_positive_definite(mat), False)


def test_is_not_positive_definite_rank_deficient():
    """Check that non-positive definite matrix returns False."""
    c_var = 1 / np.sqrt(3)
    gram = np.array(
        [
            [1, c_var, c_var, c_var],
            [c_var, 1, c_var * 1j, (1 + c_var * 1j) / 2],
            [c_var, -c_var * 1j, 1, (1 - c_var * 1j) / 2],
            [c_var, (1 - c_var * 1j) / 2, (1 + c_var * 1j) / 2, 1],
        ]
    )
    np.testing.assert_equal(is_positive_definite(gram), False)


def test_is_positive_definite_not_hermitian():
    """Input must be a Hermitian matrix."""
    mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    np.testing.assert_equal(is_positive_definite(mat), False)


def test_is_positive_definite_blas_accuracy():
    """Mathematically PSD matrix is identified as such regardless of BLAS implementation."""
    rng = np.random.default_rng(100)
    dim = 6
    x = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim)) + 0.1 * np.eye(dim)
    np.testing.assert_equal(is_positive_definite(x @ x.conj().T + 0.1 * np.eye(dim)), True)
