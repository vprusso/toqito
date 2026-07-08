"""Converts a Cvxpy expression into a np.array."""

import numpy as np
from cvxpy.expressions.expression import Expression


def expr_as_np_array(cvx_expr: Expression) -> np.ndarray:
    """Convert cvxpy expression into a numpy array.

    Args:
        cvx_expr: The cvxpy expression to be converted.

    Returns:
        The numpy array of the cvxpy expression.

    Examples:
        Convert a 2-by-2 cvxpy variable into a numpy array whose entries are the
        scalar cvxpy expressions indexing into it:

        ```python exec="1" source="above" result="text"
        import cvxpy
        from toqito.matrix_ops import expr_as_np_array

        x_var = cvxpy.Variable((2, 2), name="X")
        arr = expr_as_np_array(x_var)
        print(arr.shape)
        print(arr[0, 1])
        ```

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
