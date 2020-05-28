"""Tests for helper."""
import cvxpy
import numpy as np

from toqito.helper import update_odometer
from toqito.helper import expr_as_np_array
from toqito.helper import np_array_as_expr


def test_np_array_as_expr():
    """Ensure return type is CVX object."""
    test_input_mat = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])

    res_mat = np_array_as_expr(test_input_mat)
    np.testing.assert_equal(isinstance(res_mat, cvxpy.atoms.affine.vstack.Vstack), True)


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


def test_update_odometer_0_0():
    """Update odometer from [2, 2] to [0, 0]."""
    vec = np.array([2, 2])
    upper_lim = np.array([3, 2])
    res = update_odometer(vec, upper_lim)

    bool_mat = np.isclose([0, 0], res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_update_odometer_0_1():
    """Update odometer from [0, 0] to [0, 1]."""
    vec = np.array([0, 0])
    upper_lim = np.array([3, 2])
    res = update_odometer(vec, upper_lim)

    bool_mat = np.isclose([0, 1], res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_update_odometer_1_0():
    """Update odometer from [0, 1] to [1, 0]."""
    vec = np.array([0, 1])
    upper_lim = np.array([3, 2])
    res = update_odometer(vec, upper_lim)

    bool_mat = np.isclose([1, 0], res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_update_odometer_2_0():
    """Update odometer from [1, 1] to [2, 0]."""
    vec = np.array([1, 1])
    upper_lim = np.array([3, 2])
    res = update_odometer(vec, upper_lim)

    bool_mat = np.isclose([2, 0], res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_update_odometer_2_1():
    """Update odometer from [2, 0] to [2, 1]."""
    vec = np.array([2, 0])
    upper_lim = np.array([3, 2])
    res = update_odometer(vec, upper_lim)

    bool_mat = np.isclose([2, 1], res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_update_odometer_2_2():
    """Update odometer from [2, 1] to [0, 0]."""
    vec = np.array([2, 1])
    upper_lim = np.array([3, 2])
    res = update_odometer(vec, upper_lim)

    bool_mat = np.isclose([0, 0], res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_update_odometer_empty():
    """Return `None` if empty lists are provided."""
    vec = np.array([])
    upper_lim = np.array([])
    res = update_odometer(vec, upper_lim)

    bool_mat = np.isclose([], res)
    np.testing.assert_equal(np.all(bool_mat), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
