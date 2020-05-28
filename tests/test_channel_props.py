"""Tests for channel_props."""
import numpy as np

from toqito.channel_props import is_completely_positive
from toqito.channel_props import is_herm_preserving
from toqito.channel_props import is_positive

from toqito.perms import swap_operator
from toqito.channels import depolarizing


def test_is_completely_positive_kraus_false():
    """Verify non-completely positive channel as Kraus ops as False."""
    unitary_mat = np.array([[1, 1], [-1, 1]]) / np.sqrt(2)
    kraus_ops = [[np.identity(2), np.identity(2)], [unitary_mat, -unitary_mat]]

    np.testing.assert_equal(is_completely_positive(kraus_ops), False)


def test_is_completely_positive_choi_true():
    """
    Verify that the Choi matrix of the depolarizing map is completely
    positive.
    """
    np.testing.assert_equal(is_completely_positive(depolarizing(2)), True)


def test_is_herm_preserving_kraus_true():
    """Verify Hermitian-preserving channel as Kraus ops as True."""
    unitary_mat = np.array([[1, 1], [-1, 1]]) / np.sqrt(2)
    kraus_ops = [[np.identity(2), np.identity(2)], [unitary_mat, -unitary_mat]]

    np.testing.assert_equal(is_herm_preserving(kraus_ops), True)


def test_is_herm_preserving_kraus_false():
    """Verify non-Hermitian-preserving channel as Kraus ops as False."""
    unitary_mat = np.array([[1, 1], [-1, 1]]) / np.sqrt(2)
    kraus_ops = [[np.identity(2), unitary_mat], [unitary_mat, -unitary_mat]]

    np.testing.assert_equal(is_herm_preserving(kraus_ops), False)


def test_is_herm_preserving_choi_true():
    """
    The swap operator is the Choi matrix of the (Hermitian-preserving)
    transpose map.
    """
    choi_mat = swap_operator(3)
    np.testing.assert_equal(is_herm_preserving(choi_mat), True)


def test_is_herm_preserving_non_square():
    """Verify non-square Choi matrix returns False."""
    non_square_mat = np.array([[1, 2, 3], [4, 5, 6]])
    np.testing.assert_equal(is_herm_preserving(non_square_mat), False)


def test_is_positive_kraus_false():
    """Verify non-completely positive channel as Kraus ops as False."""
    unitary_mat = np.array([[1, 1], [-1, -1]]) / np.sqrt(2)
    kraus_ops = [[np.identity(2), np.identity(2)], [unitary_mat, -unitary_mat]]

    np.testing.assert_equal(is_positive(kraus_ops), False)


def test_is_positive_choi_true():
    """
    Verify that the Choi matrix of the depolarizing map is positive.
    """
    np.testing.assert_equal(is_positive(depolarizing(4)), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
