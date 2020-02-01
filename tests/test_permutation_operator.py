"""Tests for permutation_operator function."""
import unittest
import numpy as np

from scipy.sparse import csr_matrix
from toqito.perms.permutation_operator import permutation_operator


class TestPermutationOperator(unittest.TestCase):
    """Unit test for permutation_operator."""

    def test_standard_swap(self):
        """Generates the standard swap operator on two qubits."""
        expected_res = np.array([[1, 0, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 0, 1]])

        res = permutation_operator(2, [2, 1])

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)

    def test_standard_swap_list_dim(self):
        """Generates the standard swap operator on two qubits."""
        expected_res = np.array([[1, 0, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 0, 1]])

        res = permutation_operator([2, 2], [2, 1])

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)

    def test_sparse_option(self):
        """Sparse swap operator on two qutrits."""
        res = permutation_operator(3, [2, 1], False, True)

        self.assertEqual(res[0][0], 1)


if __name__ == '__main__':
    unittest.main()
