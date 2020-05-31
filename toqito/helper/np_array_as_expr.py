"""np.array object as cvxpy expression object."""
import numpy as np
from cvxpy import bmat
from cvxpy.expressions.expression import Expression


def np_array_as_expr(np_arr: [np.ndarray]) -> Expression:
    """
    Convert numpy array into a cvxpy expression.

    :param np_arr: The numpy array to be converted.
    :return: The cvxpy expression of the numpy array.
    """
    as_list = np_arr.tolist()
    expr = bmat(as_list)
    return expr
