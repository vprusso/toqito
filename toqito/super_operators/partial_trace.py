"""Computes the partial trace of a matrix."""
from typing import Union, List
import numpy as np
from scipy.sparse import issparse, csr_matrix
from skimage.util.shape import view_as_blocks
from toqito.perms.permute_systems import permute_systems
from toqito.helper.cvx_helper import expr_as_np_array, np_array_as_expr
from cvxpy.expressions.expression import Expression


def partial_trace_cvx(rho, sys=None, dim=None):
    """
    :param rho:
    :param sys:
    :param dim:
    :return:
    """
    rho_np = expr_as_np_array(rho)
    traced_rho = partial_trace(rho_np, sys, dim)
    traced_rho = np_array_as_expr(traced_rho)
    return traced_rho


def partial_trace(input_mat: np.ndarray,
                  sys: Union[int, List[int]] = 2,
                  dim: Union[int, List[int]] = None):
    """
    Computes the partial trace of a matrix.

    :param input_mat: A square matrix.
    :param sys:
    :param dim:

    Gives the partial trace of the matrix X, where the dimensions of the
    (possibly more than 2) subsystems are given by the vector DIM and the
    subsystems to take the trace on are given by the scalar or vector SYS.
    """
    if dim is None:
        dim = np.array([np.round(np.sqrt(len(input_mat)))])
    if isinstance(dim, int):
        dim = np.array([dim])
    if isinstance(dim, list):
        dim = np.array(dim)

    if sys is None:
        sys = 2

    num_sys = len(dim)

    # Allow the user to enter a single number for dim.
    if num_sys == 1:
        dim = np.array([dim[0], len(input_mat)/dim[0]])
        if np.abs(dim[1] - np.round(dim[1])) >= 2 * \
                len(input_mat) * np.finfo(float).eps:
            msg = """
                InvalidDim: If `dim` is a scalar, `dim` must evenly divide 
                `len(input_mat)`.
            """
            raise ValueError(msg)
        dim[1] = np.round(dim[1])
        num_sys = 2

    is_sparse = issparse(input_mat)
    prod_dim = np.prod(dim)
    if isinstance(sys, list):
        prod_dim_sys = np.prod(dim[sys])
    elif isinstance(sys, int):
        prod_dim_sys = np.prod(dim[sys-1])
    else:
        msg = """
            InvalidSys: The variable `sys` must either be of type int of a
            list of ints.
        """
        raise ValueError(msg)
    
    sub_prod = prod_dim / prod_dim_sys
    sub_sys_vec = prod_dim * np.ones(int(sub_prod)) / sub_prod

    if isinstance(sys, int):
        sys = [sys]
    s2 = sys
    set_diff = list(set(list(range(1, num_sys+1))) - set(s2))

    perm = set_diff
    perm.extend(sys)

    a_mat = permute_systems(input_mat, perm, dim)

    x = np.reshape(a_mat, [int(sub_sys_vec[0]), int(sub_prod), int(sub_sys_vec[0]), int(sub_prod)], order="F")
    y = x.transpose((1, 3, 0, 2))
    z = np.reshape(y, [int(sub_prod), int(sub_prod), int(sub_sys_vec[0]**2)], order="F")

    q = z[:, :, list(range(0, int(sub_sys_vec[0]**2), int(sub_sys_vec[0]+1)))]

    Xpt = np.sum(q, axis=2)

    return Xpt

