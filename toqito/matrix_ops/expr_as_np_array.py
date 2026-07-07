"""Converts a Cvxpy expression into a np.array."""

import numpy as np
from cvxpy.expressions.expression import Expression


def expr_as_np_array(cvx_expr: Expression) -> np.ndarray:
    """Convert cvxpy expression into a numpy array.

    Args:
        cvx_expr: The cvxpy expression to be converted.

    Examples:
        Convert a 2x2 CVXPY variable into a NumPy array.

        ```python exec="1" source="above"
        import cvxpy as cp

        from toqito.matrix_ops import expr_as_np_array

        x = cp.Variable((2, 2))
        arr = expr_as_np_array(x)

        print(arr.shape)
        ```

    Returns:
        The numpy array of the cvxpy expression.

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
