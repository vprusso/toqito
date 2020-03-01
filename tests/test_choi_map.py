"""Tests for choi_map function."""
import unittest
import numpy as np

from toqito.super_operators.choi_map import choi_map


class TestChoiMap(unittest.TestCase):
    """Unit test for choi_map."""

    def test_standard_choi_map(self):
        """The standard Choi map."""
        expected_res = np.array([[1, 0, 0, 0, -1, 0, 0, 0, -1],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 1, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 1, 0, 0, 0, 0, 0],
                                 [-1, 0, 0, 0, 1, 0, 0, 0, -1],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 1, 0],
                                 [-1, 0, 0, 0, -1, 0, 0, 0, 1]])

        res = choi_map()

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)

    def test_reduction_map(self):
        """
        The reduction map is the map R defined by: R(X) = Tr(X)I - X.

        The reduction map is the Choi map that arises when a = 0, b = c = 1.
        """
        expected_res = np.array([[0, 0, 0, 0, -1, 0, 0, 0, -1],
                                 [0, 1, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 1, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 1, 0, 0, 0, 0, 0],
                                 [-1, 0, 0, 0, 0, 0, 0, 0, -1],
                                 [0, 0, 0, 0, 0, 1, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 1, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 1, 0],
                                 [-1, 0, 0, 0, -1, 0, 0, 0, 0]])

        res = choi_map(0, 1, 1)

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)


if __name__ == '__main__':
    unittest.main()
