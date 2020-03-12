"""Tests for partial_transpose function."""
import unittest
import numpy as np

from toqito.state.states.bell import bell
from toqito.super_operators.partial_transpose import partial_transpose


class TestPartialTranspose(unittest.TestCase):
    """Unit test for partial_transpose."""

    def test_partial_transpose(self):
        """
        Default partial_transpose.

        By default, the partial_transpose function performs the transposition
        on the second subsystem.
        """
        test_input_mat = np.arange(1, 17).reshape(4, 4)

        expected_res = np.array([[1, 5, 3, 7],
                                 [2, 6, 4, 8],
                                 [9, 13, 11, 15],
                                 [10, 14, 12, 16]])

        res = partial_transpose(test_input_mat)

        bool_mat = np.isclose(expected_res, res)
        self.assertEqual(np.all(bool_mat), True)

    def test_partial_transpose_sys(self):
        """
        Default partial transpose `sys` argument.

        By specifying the `sys` argument, you can perform the transposition on
        the first subsystem instead:
        """
        test_input_mat = np.arange(1, 17).reshape(4, 4)

        expected_res = np.array([[1, 2, 9, 10],
                                 [5, 6, 13, 14],
                                 [3, 4, 11, 12],
                                 [7, 8, 15, 16]])

        res = partial_transpose(test_input_mat, 1)

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)

    def test_partial_transpose_sys_vec(self):
        """Partial transpose on matrix with `sys` defined as vector."""
        test_input_mat = np.arange(1, 17).reshape(4, 4)

        expected_res = np.array([[1, 5, 9, 13],
                                 [2, 6, 10, 14],
                                 [3, 7, 11, 15],
                                 [4, 8, 12, 16]])

        res = partial_transpose(test_input_mat, [1, 2])

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)

    def test_partial_transpose_sys_vec_dim_vec(self):
        """Variables `sys` and `dim` defined as vector."""
        test_input_mat = np.arange(1, 17).reshape(4, 4)

        expected_res = np.array([[1, 5, 9, 13],
                                 [2, 6, 10, 14],
                                 [3, 7, 11, 15],
                                 [4, 8, 12, 16]])

        res = partial_transpose(test_input_mat, [1, 2], [2, 2])

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)

    def test_partial_transpose_norm_diff(self):
        """
        Apply partial transpose to first and second subsystem.

        Applying the transpose to both the first and second subsystems results
        in the standard transpose of the matrix.
        """
        test_input_mat = np.arange(1, 17).reshape(4, 4)
        res = np.linalg.norm(partial_transpose(
            test_input_mat, [1, 2]) - test_input_mat.conj().T)
        expected_res = 0

        self.assertEqual(np.isclose(res, expected_res), True)

    def test_partial_transpose_16_by_16(self):
        """Partial transpose on a 16-by-16 matrix."""
        test_input_mat = np.arange(1, 257).reshape(16, 16)
        res = partial_transpose(test_input_mat, [1, 3], [2, 2, 2, 2])
        first_expected_row = np.array([1, 2, 33, 34, 5, 6, 37, 38, 129, 130,
                                       161, 162, 133, 134, 165, 166])

        first_expected_col = np.array([1, 17, 3, 19, 65, 81, 67, 83, 9, 25, 11,
                                       27, 73, 89, 75, 91])

        self.assertEqual(np.allclose(res[0, :], first_expected_row), True)
        self.assertEqual(np.allclose(res[:, 0], first_expected_col), True)

    def test_bell_state_pt(self):
        """Test partial transpose on a Bell state."""
        rho = bell(2) * bell(2).conj().T
        expected_res = np.array([[0, 0, 0, 1/2],
                                 [0, 1/2, 0, 0],
                                 [0, 0, 1/2, 0],
                                 [1/2, 0, 0, 0]])
        res = partial_transpose(rho)
        self.assertEqual(np.allclose(res, expected_res), True)

    def test_non_square_matrix(self):
        """Matrix must be square."""
        with self.assertRaises(ValueError):
            test_input_mat = np.array([[1, 2, 3, 4],
                                       [5, 6, 7, 8],
                                       [13, 14, 15, 16]])
            partial_transpose(test_input_mat)

    def test_non_square_matrix_2(self):
        """Matrix must be square."""
        with self.assertRaises(ValueError):
            rho = np.array([[1, 2, 3, 4],
                            [5, 6, 7, 8],
                            [9, 10, 11, 12]])
            partial_transpose(rho, 2, [2])


if __name__ == '__main__':
    unittest.main()
