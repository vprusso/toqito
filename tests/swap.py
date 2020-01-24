from toqito.helper.swap import swap

import itertools
import unittest
import numpy as np


class TestSwap(unittest.TestCase):
    """Unit test for swap."""

    def test_m2_m2(self):
        """
        """
        X = np.array([[1, 5, 9, 13],
                      [2, 6, 10, 14],
                      [3, 7, 11, 15],
                      [4, 8, 12, 16]])

        expected_res = np.array([[1, 9, 5, 13],
                                 [3, 11, 7, 15],
                                 [2, 10, 6, 14],
                                 [4, 12, 8, 16]])

        res = swap(X)

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(all(x == 1 for x in itertools.chain(*bool_mat)), True)


if __name__ == '__main__':
    unittest.main()
