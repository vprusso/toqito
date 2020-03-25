"""Tests for cvx_helper function."""
import unittest
import cvxpy
import numpy as np

from toqito.helper.cvxpy_helper import expr_as_np_array, np_array_as_expr


class TestCVXPYHelper(unittest.TestCase):
    """Unit test for cvx_helper."""

    def test_np_array_as_expr(self):
        """Ensure return type is CVX object."""
        test_input_mat = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])

        res_mat = np_array_as_expr(test_input_mat)
        self.assertEqual(isinstance(res_mat, cvxpy.atoms.affine.vstack.Vstack), True)

    def test_expr_as_np_array(self):
        """Ensure return type is numpy object."""
        expr = cvxpy.bmat([[1, 2], [3, 4]])

        res_mat = expr_as_np_array(expr)
        self.assertEqual(isinstance(res_mat, np.ndarray), True)

    def test_expr_as_np_array_scalar(self):
        """Ensure return type is numpy object for scalar expression."""
        cvx_var = cvxpy.Variable()
        self.assertEqual(isinstance(expr_as_np_array(cvx_var), np.ndarray), True)

    def test_expr_as_np_array_vector(self):
        """Ensure return type is numpy object for vector expression."""
        cvx_var = cvxpy.Parameter(5)
        self.assertEqual(isinstance(expr_as_np_array(cvx_var), np.ndarray), True)


if __name__ == "__main__":
    unittest.main()
