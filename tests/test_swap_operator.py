from toqito.helper.swap_operator import swap_operator

import itertools
import unittest
import numpy as np


class TestSwapOperator(unittest.TestCase):
    """Unit test for swap_operator."""

    def test_swap_operator_num(self):
        """Tests swap operator when argument is number."""
        expected_res = np.array([[1, 0, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 0, 1]])

        res = swap_operator(2)

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(all(x == 1 for x in itertools.chain(*bool_mat)), True)

    def test_swap_operator_vec_dims(self):
        """Tests swap operator when argument is vector of dims."""
        expected_res = np.array([[1, 0, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 0, 1]])

        res = swap_operator([2, 2])

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(all(x == 1 for x in itertools.chain(*bool_mat)), True)


if __name__ == '__main__':
    unittest.main()
