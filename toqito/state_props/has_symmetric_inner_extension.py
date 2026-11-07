"""Determine whether a bipartite operator has a symmetric inner extension."""

import math

import cvxpy
import numpy as np
from scipy.special import roots_jacobi

from toqito.matrix_ops._vec_maps import _apply_vec_map, _vec_kron_identity
from toqito.matrix_props import is_positive_semidefinite
from toqito.state_opt._symmetric_extension import SymmetricExtension


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

    Examples:
        The two-qubit maximally mixed state \(\mathbb{I}_4 / 4\) is separable, and the
        inner extension detects it at the default level.

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.state_props import has_symmetric_inner_extension

        print(has_symmetric_inner_extension(np.identity(4) / 4))
        ```

        A Bell state is entangled, so it lies outside the cone.

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.states import bell
        from toqito.state_props import has_symmetric_inner_extension

        rho = bell(0) @ bell(0).conj().T

        print(has_symmetric_inner_extension(rho))
        ```

        Note the asymmetry: `True` proves separability, while `False` proves nothing --
        the state may be separable but outside this inner approximation.

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

    # The extension sigma lives on X (x) Sym^level(Y); its marginals on X (x) Y and on X enter the affine image
    # through one sparse map each, so the whole image is a single linear map of the variable.
    extension = SymmetricExtension(dim_x, dim_y, level)
    traced_ab = extension.partial_trace_map(list(range(2, level + 1)))
    traced_a_times_identity = _vec_kron_identity(dim_x, dim_y) @ extension.partial_trace_map(list(range(1, level + 1)))

    if ppt:
        jacobi_degree = dim_y - 2
        if jacobi_degree <= 0:
            eta = 1.0
        else:
            roots, _weights = roots_jacobi(jacobi_degree, level % 2, math.floor(level / 2) + 1)
            eta = float(np.min(1.0 - roots))
        eta *= dim_y / (2 * (dim_y - 1))
        image_map = (1.0 - eta) * traced_ab + (eta / dim_y) * traced_a_times_identity
    else:
        image_map = (level * traced_ab + traced_a_times_identity) / (level + dim_y)

    constraints = [extension.variable >> 0, _apply_vec_map(image_map, extension.variable, dim_x * dim_y) == rho]
    if ppt:
        constraints.append(extension.partial_transpose(list(range(math.ceil(level / 2) + 1))) >> 0)

    problem = cvxpy.Problem(cvxpy.Minimize(0), constraints)
    problem.solve(solver=cvxpy.SCS, eps_abs=tol, eps_rel=tol, max_iters=100_000, verbose=False)

    return problem.status in {"optimal", "optimal_inaccurate"}
