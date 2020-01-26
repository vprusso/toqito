from toqito.super_operators.apply_map import apply_map
from toqito.helper.swap_operator import swap_operator

import itertools
import unittest
import numpy as np


class TestApplyMap(unittest.TestCase):
    """Unit test for apply_map."""

    def test_apply_map_choi(self):
        """
        The swap operator is the Choi matrix of the transpose map.
        Thus, the following test is (a slow and ugly) way of computing
        the transpose of a matrix.
        """
        test_input_mat = np.array([[1, 4, 7],
                                   [2, 5, 8],
                                   [3, 6, 9]])

        expected_res = np.array([[1, 2, 3],
                                 [4, 5, 6],
                                 [7, 8, 9]])

        res = apply_map(test_input_mat, swap_operator(3))

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(all(x == 1 for x in itertools.chain(*bool_mat)), True)

    def test_apply_map_kraus(self):
        """
        The following test computes PHI(X) where X = [[1, 2], [3, 4]] and 
        where PHI is the superoperator defined by:
            Phi(X) = [[1,5],[1,0],[0,2]] X [[0,1][2,3][4,5]].conj().T - 
                     [[1,0],[0,0],[0,1]] X [[0,0][1,1],[0,0]].conj().T
        """
        test_input_mat = np.array([[1, 2],
                                   [3, 4]])

        K1 = np.array([[1, 5], [1, 0], [0, 2]])
        K2 = np.array([[0, 1], [2, 3], [4, 5]])
        K3 = np.array([[-1, 0], [0, 0], [0, -1]])
        K4 = np.array([[0, 0], [1, 1], [0, 0]])

        expected_res = np.array([[22, 95, 174],
                                 [2, 8, 14],
                                 [8, 29, 64]])

        res = apply_map(test_input_mat, [[K1, K2], [K3, K4]])

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(all(x == 1 for x in itertools.chain(*bool_mat)), True)


if __name__ == '__main__':
    unittest.main()
