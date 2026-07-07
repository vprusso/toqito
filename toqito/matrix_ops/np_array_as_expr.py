"""Converts a np.array object to cvxpy expression object."""

import numpy as np
from cvxpy import bmat
from cvxpy.expressions.expression import Expression


def np_array_as_expr(np_arr: np.ndarray) -> Expression:
    """Convert numpy array into a cvxpy expression.

    Args:
        np_arr: The numpy array to be converted.

    Returns:
        The cvxpy expression of the numpy array.

    Examples:
        Convert a 2-by-2 numpy array into a constant cvxpy expression:

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.matrix_ops import np_array_as_expr

        arr = np.array([[1, 2], [3, 4]])
        expr = np_array_as_expr(arr)
        print(expr.shape)
        print(expr.value)
        ```

    """
    as_list = np_arr.tolist()
    expr = bmat(as_list)
    return expr
