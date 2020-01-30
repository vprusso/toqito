"""Tests for iden function."""
import unittest
import numpy as np

from toqito.helper.iden import iden


class TestIden(unittest.TestCase):
    """Unit test for iden."""

    def test_iden_full(self):
        """Full 2-dimensional identity matrix."""
        expected_res = np.array([[1, 0],
                                 [0, 1]])
        res = iden(2, False)

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)

    def test_iden_sparse(self):
        """Sparse 2-dimensional identity matrix."""
        expected_res = np.array([[1, 0], [0, 1]])
        res = iden(2, True).toarray()

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)


if __name__ == '__main__':
    unittest.main()
