"""Determine whether a bipartite operator has a symmetric inner extension."""

import math
import warnings

import cvxpy
import numpy as np
from scipy.special import roots_jacobi

from toqito.matrix_ops import partial_trace, partial_transpose
from toqito.matrix_props import is_positive_semidefinite
from toqito.perms import symmetric_projection


def has_symmetric_inner_extension(
    rho: np.ndarray,
    level: int = 2,
    dim: np.ndarray | int | None = None,
    ppt: bool = True,
    tol: float = 1e-4,
) -> bool:
    r"""Determine whether a bipartite operator lies in the symmetric inner-extension cone.

    This is the inner approximation to the separable cone introduced by
    Navascués, Owari, and Plenio [@navascues2009complete]. If this function
    returns ``True``, then the input operator is separable. If it returns
    ``False``, no conclusion can be drawn.

    The implementation closely follows QETLAB's
    ``SymmetricInnerExtension`` routine, solving an SDP over the bosonic
    symmetric subspace of the repeated second subsystem.

    Args:
        rho: A positive semidefinite bipartite operator.
        level: Number of copies of the second subsystem in the extension.
            Must be at least 2.
        dim: Local dimensions. If ``None``, infer an equal bipartite split.
        ppt: Whether to impose the PPT inner-extension variant.
        tol: Numerical tolerance used by the SDP solver.

    Returns:
        ``True`` if ``rho`` lies in the inner cone at the requested level.

    Raises:
        ValueError: If the dimensions are incompatible or ``level < 2``.

    """
    if level < 2:
        raise ValueError("`level` must be an integer >= 2 for symmetric inner extensions.")

    len_mat = rho.shape[1]

    if dim is None:
        dim_val = int(np.round(np.sqrt(len_mat)))
    elif isinstance(dim, int):
        dim_val = dim
    else:
        dim_val = None

    if dim_val is not None:
        dim_arr = np.array([dim_val, len_mat / dim_val])
        if np.abs(dim_arr[1] - np.round(dim_arr[1])) >= 2 * len_mat * np.finfo(float).eps:
            raise ValueError("If `dim` is a scalar, it must evenly divide the length of the matrix.")
        dim_arr[1] = int(np.round(dim_arr[1]))
    else:
        dim_arr = np.array(dim)

    dim_arr = dim_arr.astype(int)
    dim_x, dim_y = int(dim_arr[0]), int(dim_arr[1])

    if min(dim_x, dim_y) == 1:
        return is_positive_semidefinite(rho)

    sdp_dim = np.array([dim_x] + [dim_y] * level, dtype=int)
    sym_basis = symmetric_projection(dim_y, level, partial=True)
    projector = np.kron(np.eye(dim_x), sym_basis)
    reduced_dim = dim_x * sym_basis.shape[1]

    sigma_reduced = cvxpy.Variable((reduced_dim, reduced_dim), hermitian=True)
    sigma = projector @ sigma_reduced @ projector.conj().T

    traced_ab = partial_trace(sigma, list(range(2, level + 1)), sdp_dim)
    traced_a = partial_trace(sigma, list(range(1, level + 1)), sdp_dim)

    if ppt:
        jacobi_degree = dim_y - 2
        if jacobi_degree <= 0:
            eta = 1.0
        else:
            roots, _weights = roots_jacobi(jacobi_degree, level % 2, math.floor(level / 2) + 1)
            eta = float(np.min(1.0 - roots))
        eta *= dim_y / (2 * (dim_y - 1))
        affine_image = (1.0 - eta) * traced_ab + eta * cvxpy.kron(traced_a, np.eye(dim_y)) / dim_y
    else:
        affine_image = traced_ab * level / (level + dim_y) + cvxpy.kron(traced_a, np.eye(dim_y)) / (level + dim_y)

    constraints = [sigma_reduced >> 0, affine_image == rho]

    if ppt:
        ppt_systems = list(range(math.ceil(level / 2) + 1))
        constraints.append(partial_transpose(sigma, ppt_systems, sdp_dim) >> 0)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Constraint.*subexpressions")
        problem = cvxpy.Problem(cvxpy.Minimize(0), constraints)
        problem.solve(solver=cvxpy.SCS, eps_abs=tol, eps_rel=tol, max_iters=100_000, verbose=False)

    return problem.status in {"optimal", "optimal_inaccurate"}
