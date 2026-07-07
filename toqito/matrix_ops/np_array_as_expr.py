"""Converts a np.array object to cvxpy expression object."""

import numpy as np
from cvxpy import bmat
from cvxpy.expressions.expression import Expression


def np_array_as_expr(np_arr: np.ndarray) -> Expression:
    """Convert numpy array into a cvxpy expression.

    Args:
        np_arr: The numpy array to be converted.

    Examples:
        Convert a CVXPY variable to a NumPy array and back into a CVXPY
        expression.

        ```python exec="1" source="above"
        import cvxpy as cp

        from toqito.matrix_ops import expr_as_np_array, np_array_as_expr

        x = cp.Variable((2, 2))
        arr = expr_as_np_array(x)
        expr = np_array_as_expr(arr)

        print(expr.shape)
        ```

    Returns:
        The cvxpy expression of the numpy array.

    """
    as_list = np_arr.tolist()
    expr = bmat(as_list)
    return expr
