from toqito.super_operators.partial_transpose import partial_transpose

import unittest
import numpy as np


class TestPartialTranspose(unittest.TestCase):
    """Unit test for partial_transpose."""

    def test_partial_transpose(self):
        """
        By default, the partial_transpose function performs the transposition
        on the second subsystem.
        """
        test_input_mat = np.array([[1, 2, 3, 4],
                                   [5, 6, 7, 8],
                                   [9, 10, 11, 12],
                                   [13, 14, 15, 16]])

        expected_res = np.array([[1, 5, 3, 7],
                                 [2, 6, 4, 8],
                                 [9, 13, 11, 15],
                                 [10, 14, 12, 16]])

        res = partial_transpose(test_input_mat)

        bool_mat = np.isclose(expected_res, res)
        self.assertEqual(np.all(bool_mat), True)

    def test_partial_transpose_sys(self):
        """
        By specifying the SYS argument, you can perform the transposition on
        the first subsystem instead:
        """
        test_input_mat = np.array([[1, 2, 3, 4],
                                   [5, 6, 7, 8],
                                   [9, 10, 11, 12],
                                   [13, 14, 15, 16]])

        expected_res = np.array([[1, 2, 9, 10],
                                 [5, 6, 13, 14],
                                 [3, 4, 11, 12],
                                 [7, 8, 15, 16]])

        res = partial_transpose(test_input_mat, 1)

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)


if __name__ == '__main__':
    unittest.main()
