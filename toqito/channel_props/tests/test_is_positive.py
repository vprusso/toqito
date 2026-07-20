"""Tests for is_positive."""

import numpy as np

from toqito.channel_props import is_completely_positive, is_positive
from toqito.channels import depolarizing
from toqito.perms import swap_operator


def test_is_positive_kraus_false():
    """A map that sends the PSD state |0><0| to an indefinite operator is not positive."""
    unitary_mat = np.array([[1, 1], [-1, -1]]) / np.sqrt(2)
    kraus_ops = [[np.identity(2), np.identity(2)], [unitary_mat, -unitary_mat]]

    np.testing.assert_equal(is_positive(kraus_ops), False)


def test_is_positive_choi_true():
    """Verify that the Choi matrix of the depolarizing map is positive."""
    np.testing.assert_equal(is_positive(depolarizing(4)), True)


def test_is_positive_transpose_not_completely_positive():
    """The transpose map (Choi = swap operator) is positive but not completely positive."""
    transpose_choi = swap_operator(2)
    np.testing.assert_equal(is_positive(transpose_choi), True)
    np.testing.assert_equal(is_completely_positive(transpose_choi), False)


def test_is_positive_non_hermitian_choi_false():
    """A non-Hermitian Choi matrix cannot come from a positive (Hermiticity-preserving) map."""
    non_hermitian = np.array([[1, 2, 3, 4], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=complex)
    np.testing.assert_equal(is_positive(non_hermitian), False)
