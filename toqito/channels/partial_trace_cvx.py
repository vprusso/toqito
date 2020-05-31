"""The partial trace on CVX objects."""
from toqito.channels import partial_trace
from toqito.helper import expr_as_np_array, np_array_as_expr


def partial_trace_cvx(rho, sys=None, dim=None):
    """
    Perform the partial trace on a :code:`cvxpy` variable.

    Adapted from [CVXPtrace]_.

    References
    ==========
    .. [CVXPtrace] Partial trace for CVXPY variables
        https://github.com/cvxgrp/cvxpy/issues/563

    :param rho: A square matrix.
    :param sys: Scalar or vector specifying the size of the subsystems.
    :param dim: Dimension of the subsystems. If :code:`None`, all dimensions
                are assumed to be equal.
    :return: The partial trace of matrix :code:`input_mat`.
    """
    rho_np = expr_as_np_array(rho)
    traced_rho = partial_trace(rho_np, sys, dim)
    traced_rho = np_array_as_expr(traced_rho)
    return traced_rho
