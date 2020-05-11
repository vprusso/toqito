"""Tests for is_herm_preserving function."""
import unittest
import numpy as np

from toqito.channel_props import is_herm_preserving
from toqito.perms import swap_operator


class TestIsHermPreserving(unittest.TestCase):

    """Unit test for is_herm_preserving."""

    def test_is_herm_preserving_kraus_true(self):
        """Verify Hermitian-preserving channel as Kraus ops as True."""
        unitary_mat = np.array([[1, 1], [-1, 1]]) / np.sqrt(2)
        kraus_ops = [[np.identity(2), np.identity(2)], [unitary_mat, -unitary_mat]]

        self.assertEqual(is_herm_preserving(kraus_ops), True)

    def test_is_herm_preserving_kraus_false(self):
        """Verify non-Hermitian-preserving channel as Kraus ops as False."""
        unitary_mat = np.array([[1, 1], [-1, 1]]) / np.sqrt(2)
        kraus_ops = [[np.identity(2), unitary_mat], [unitary_mat, -unitary_mat]]

        self.assertEqual(is_herm_preserving(kraus_ops), False)

    def test_is_herm_preserving_choi_true(self):
        """
        The swap operator is the Choi matrix of the (Hermitian-preserving)
        transpose map.
        """
        choi_mat = swap_operator(3)
        self.assertEqual(is_herm_preserving(choi_mat), True)

    def test_is_herm_preserving_non_square(self):
        """Verify non-square Choi matrix returns False."""
        non_square_mat = np.array([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(is_herm_preserving(non_square_mat), False)


if __name__ == "__main__":
    unittest.main()
