"""Test expr_as_np_array."""
import cvxpy
import numpy as np

from toqito.helper import expr_as_np_array


def test_expr_as_np_array():
    """Ensure return type is numpy object."""
    expr = cvxpy.bmat([[1, 2], [3, 4]])

    res_mat = expr_as_np_array(expr)
    np.testing.assert_equal(isinstance(res_mat, np.ndarray), True)


def test_expr_as_np_array_scalar():
    """Ensure return type is numpy object for scalar expression."""
    cvx_var = cvxpy.Variable()
    np.testing.assert_equal(isinstance(expr_as_np_array(cvx_var), np.ndarray), True)


def test_expr_as_np_array_vector():
    """Ensure return type is numpy object for vector expression."""
    cvx_var = cvxpy.Parameter(5)
    np.testing.assert_equal(isinstance(expr_as_np_array(cvx_var), np.ndarray), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
