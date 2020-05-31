"""cvxpy expression as np.array."""
import numpy as np
from cvxpy.expressions.expression import Expression


def expr_as_np_array(cvx_expr: Expression) -> np.ndarray:
    """
    Convert cvxpy expression into a numpy array.

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
