"""Tests for swap function."""
import itertools
import unittest
import numpy as np

from toqito.perms.swap import swap


class TestSwap(unittest.TestCase):
    """Unit test for swap."""

    def test_swap_matrix(self):
        """Tests swap operation on matrix."""
        test_mat = np.array([[1, 5, 9, 13],
                             [2, 6, 10, 14],
                             [3, 7, 11, 15],
                             [4, 8, 12, 16]])

        expected_res = np.array([[1, 9, 5, 13],
                                 [3, 11, 7, 15],
                                 [2, 10, 6, 14],
                                 [4, 12, 8, 16]])

        res = swap(test_mat)

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(all(x == 1 for x in itertools.chain(*bool_mat)), True)

    def test_swap_vector(self):
        """Tests swap operation on vector."""
        test_vec = np.array([[1, 2, 3, 4]])

        expected_res = np.array([[1, 3, 2, 4]])

        res = swap(test_vec)

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(all(x == 1 for x in itertools.chain(*bool_mat)), True)


if __name__ == '__main__':
    unittest.main()
