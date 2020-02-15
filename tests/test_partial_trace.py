"""Tests for partial_trace function."""
import unittest
import numpy as np

from toqito.super_operators.partial_trace import partial_trace


class TestPartialTrace(unittest.TestCase):
    """Unit test for partial_trace."""

    def test_partial_trace(self):
        """
        By default, the partial_transpose function takes the trace over
        the second subsystem.
        """
        test_input_mat = np.array([[1, 2, 3, 4],
                                   [5, 6, 7, 8],
                                   [9, 10, 11, 12],
                                   [13, 14, 15, 16]])

        expected_res = np.array([[7, 11],
                                 [23, 27]])

        res = partial_trace(test_input_mat)

        bool_mat = np.isclose(expected_res, res)
        self.assertEqual(np.all(bool_mat), True)

    def test_partial_trace_sys(self):
        """
        By specifying the SYS argument, you can perform the partial trace
        the first subsystem instead:
        """
        test_input_mat = np.array([[1, 2, 3, 4],
                                   [5, 6, 7, 8],
                                   [9, 10, 11, 12],
                                   [13, 14, 15, 16]])

        expected_res = np.array([[12, 14],
                                 [20, 22]])

        res = partial_trace(test_input_mat, 1)

        bool_mat = np.isclose(expected_res, res)
        self.assertEqual(np.all(bool_mat), True)

    def test_partial_trace_sys_list(self):
        """
        Taking the partial trace of both the first and second subsystems
        results in the standard trace of X.
        """
        test_input_mat = np.array([[1, 2, 3, 4],
                                   [5, 6, 7, 8],
                                   [9, 10, 11, 12],
                                   [13, 14, 15, 16]])

        expected_res = 34

        res = partial_trace(test_input_mat, [1, 2])
        self.assertEqual(res, expected_res)

    def test_partial_trace_sys_int_dim_int(self):
        """
        By default, the partial_transpose function takes the trace over
        the second subsystem.
        """
        test_input_mat = np.array([[1, 2, 3, 4],
                                   [5, 6, 7, 8],
                                   [9, 10, 11, 12],
                                   [13, 14, 15, 16]])

        expected_res = np.array([[7, 11],
                                 [23, 27]])

        res = partial_trace(test_input_mat, 2, 2)

        bool_mat = np.isclose(expected_res, res)
        self.assertEqual(np.all(bool_mat), True)

    def test_partial_trace_sys_int_dim_int_2(self):
        """
        By default, the partial_transpose function takes the trace over
        the second subsystem.
        """
        test_input_mat = np.array([[1, 2, 3, 4],
                                   [5, 6, 7, 8],
                                   [9, 10, 11, 12],
                                   [13, 14, 15, 16]])

        expected_res = 34

        res = partial_trace(test_input_mat, 2, 1)

        bool_mat = np.isclose(expected_res, res)
        self.assertEqual(np.all(bool_mat), True)


if __name__ == '__main__':
    unittest.main()
