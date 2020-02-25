"""Helper functions for dealing with cvxpy objects."""
import numpy as np
from cvxpy import bmat
from cvxpy.expressions.expression import Expression


def expr_as_np_array(cvx_expr: Expression) -> np.ndarray:
    """
    Converts a cvxpy expression into a numpy array.

    :param cvx_expr: The cvxpy expression to be converted.
    :return: The numpy array of the cvxpy expression.
    """
    if cvx_expr.is_scalar():
        return np.array(cvx_expr)
    if len(cvx_expr.shape) == 1:
        return np.array(list(cvx_expr))
    # Then cvx_expr is a 2-D array.
    rows = []
    for i in range(cvx_expr.shape[0]):
        row = [cvx_expr[i, j] for j in range(cvx_expr.shape[1])]
        rows.append(row)
    arr = np.array(rows)
    return arr


def np_array_as_expr(np_arr: [np.ndarray]) -> Expression:
    """
    Converts a numpy array into a cvxpy expression.

    :param np_arr: The numpy array to be converted.
    :return: The cvxpy expression of the numpy array.
    """
    as_list = np_arr.tolist()
    expr = bmat(as_list)
    return expr
