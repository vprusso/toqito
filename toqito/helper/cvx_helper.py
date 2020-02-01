"""Helper functions for dealing with CVXPY objects."""
import cvxpy
import numpy as np
from cvxpy.expressions.expression import Expression


def expr_as_np_array(cvx_expr: Expression) -> np.ndarray:
    """
    :param cvx_expr:
    :return:
    """
    if cvx_expr.is_scalar():
        return np.array(cvx_expr)
    elif len(cvx_expr.shape) == 1:
        return np.array([v for v in cvx_expr])
    else:
        # Then cvx_expr is a 2-D array.
        rows = []
        for i in range(cvx_expr.shape[0]):
            row = [cvx_expr[i, j] for j in range(cvx_expr.shape[1])]
            rows.append(row)
        arr = np.array(rows)
        return arr


def np_array_as_expr(np_arr: [np.ndarray]) -> Expression:
    """
    :param np_arr:
    :return:
    """
    as_list = np_arr.tolist()
    expr = cvxpy.bmat(as_list)
    return expr
