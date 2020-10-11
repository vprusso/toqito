"""Kronecker product for CVXPY objects."""
from typing import Union

import numpy as np
import cvxpy
from cvxpy.expressions.expression import Expression


def cvx_kron(
    expr_1: Union[np.ndarray, Expression], expr_2: Union[np.ndarray, Expression]
) -> Expression:
    """
    Compute Kronecker product between CVXPY objects.

    By default, CVXPY does not support taking the Kronecker product when the argument on the left is
    equal to a CVXPY object and the object on the right is equal to a numpy object.

    At most one of :code:`expr_1` and :code:`b` may be CVXPY Variable objects.

    Kudos to Riley J. Murray for this function:
    https://github.com/cvxgrp/cvxpy/issues/457

    :param expr_1: 2D numpy ndarray, or a CVXPY Variable with expr_1.ndim == 2
    :param expr_2: 2D numpy ndarray, or a CVXPY Variable with expr_2.ndim == 2
    :return: The tensor product of two CVXPY expressions.
    """
    expr = np.kron(expr_1, expr_2)
    num_rows = expr.shape[0]
    rows = [cvxpy.hstack(expr[i, :]) for i in range(num_rows)]
    full_expr = cvxpy.vstack(rows)
    return full_expr
