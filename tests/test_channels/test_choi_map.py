"""Tests for choi_map function."""
import unittest
import numpy as np

from toqito.channels import choi


class TestChoiMap(unittest.TestCase):

    """Unit test for choi_map."""

    def test_standard_choi_map(self):
        """The standard Choi map."""
        expected_res = np.array(
            [
                [1, 0, 0, 0, -1, 0, 0, 0, -1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0],
                [-1, 0, 0, 0, 1, 0, 0, 0, -1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                [-1, 0, 0, 0, -1, 0, 0, 0, 1],
            ]
        )

        res = choi()

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)

    def test_reduction(self):
        """
        The reduction map is the map R defined by: R(X) = Tr(X)I - X.

        The reduction map is the Choi map that arises when a = 0, b = c = 1.
        """
        expected_res = np.array(
            [
                [0, 0, 0, 0, -1, 0, 0, 0, -1],
                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0],
                [-1, 0, 0, 0, 0, 0, 0, 0, -1],
                [0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                [-1, 0, 0, 0, -1, 0, 0, 0, 0],
            ]
        )

        res = choi(0, 1, 1)

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)


if __name__ == "__main__":
    unittest.main()
