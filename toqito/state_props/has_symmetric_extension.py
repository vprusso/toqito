"""Determine whether there exists a symmetric extension for a given quantum state."""

import cvxpy
import numpy as np

from toqito.matrix_ops import partial_trace, partial_transpose
from toqito.matrix_props import is_positive_semidefinite
from toqito.perms import symmetric_projection
from toqito.state_props.is_ppt import is_ppt


def has_symmetric_extension(
    rho: np.ndarray,
    level: int = 2,
    dim: np.ndarray | int | None = None,
    ppt: bool = True,
    tol: float = 1e-4,
) -> bool:
    r"""Determine whether there exists a symmetric extension for a given quantum state.

    For more information, see :footcite:`Doherty_2002_Distinguishing`.

    Determining whether an operator possesses a symmetric extension at some level :code:`level`
    can be used as a check to determine if the operator is entangled or not.

    This function was adapted from QETLAB.

    Examples
    ==========

    2-qubit symmetric extension:

    In :footcite:`Chen_2014_Symmetric`, it was shown that a 2-qubit state :math:`\rho_{AB}` has a
    symmetric extension if and only if

    .. math::
        \text{Tr}(\rho_B^2) \geq \text{Tr}(\rho_{AB}^2) - 4 \sqrt{\text{det}(\rho_{AB})}.

    This closed-form equation is much quicker to check than running the semidefinite program.

    .. jupyter-execute::

        import numpy as np
        from toqito.state_props import has_symmetric_extension
        from toqito.matrix_ops import partial_trace
        rho = np.array([[1, 0, 0, -1], [0, 1, 1/2, 0], [0, 1/2, 1, 0], [-1, 0, 0, 1]])
        # Show the closed-form equation holds
        np.trace(np.linalg.matrix_power(partial_trace(rho, 1), 2)) >= np.trace(rho**2) - 4 * np.sqrt(np.linalg.det(rho))

    .. jupyter-execute::

        # Now show that the `has_symmetric_extension` function recognizes this case.
        has_symmetric_extension(rho)

    Higher qubit systems:

    Consider a density operator corresponding to one of the Bell states.

    .. math::
        \rho = \frac{1}{2} \begin{pmatrix}
                            1 & 0 & 0 & 1 \\
                            0 & 0 & 0 & 0 \\
                            0 & 0 & 0 & 0 \\
                            1 & 0 & 0 & 1
                           \end{pmatrix}

    To make this state over more than just two qubits, let's construct the following state

    .. math::
        \sigma = \rho \otimes \rho.

    As the state :math:`\sigma` is entangled, there should not exist a symmetric extension at some
    level. We see this being the case for a relatively low level of the hierarchy.

    .. jupyter-execute::

        import numpy as np
        from toqito.states import bell
        from toqito.state_props import has_symmetric_extension
        rho = bell(0) @ bell(0).conj().T
        sigma = np.kron(rho, rho)
        has_symmetric_extension(sigma)


    References
    ==========
    .. footbibliography::



    :raises ValueError: If dimension does not evenly divide matrix length.
    :param rho: A matrix or vector.
    :param level: Level of the hierarchy to compute.
    :param dim: The default has both subsystems of equal dimension.
    :param ppt: If :code:`True`, this enforces that the symmetric extension must be PPT.
    :param tol: Tolerance when determining whether a symmetric extension exists.
    :return: :code:`True` if :code:`mat` has a symmetric extension; :code:`False` otherwise.

    """
    len_mat = rho.shape[1]

    # Set default dimension if none was provided.
    if dim is None:
        dim_val = int(np.round(np.sqrt(len_mat)))
    elif isinstance(dim, int):
        dim_val = dim
    else:
        dim_val = None

    # Allow the user to enter in a single integer for dimension.
    if dim_val is not None:
        dim_arr = np.array([dim_val, len_mat / dim_val])
        if np.abs(dim_arr[1] - np.round(dim_arr[1])) >= 2 * len_mat * np.finfo(float).eps:
            raise ValueError("If `dim` is a scalar, it must evenly divide the length of the matrix.")
        dim_arr[1] = int(np.round(dim_arr[1]))
    else:
        dim_arr = np.array(dim)

    dim_arr = np.int_(dim_arr)

    dim_x, dim_y = int(dim_arr[0]), int(dim_arr[1])
    # In certain situations, we don't need semidefinite programming.
    if level == 1 or len_mat <= 6 and ppt:
        if not ppt:
            # In some cases, the problem is *really* trivial.
            return is_positive_semidefinite(rho)

        # In this case, all they asked for is a 1-copy PPT symmetric extension
        # (i.e., they're asking if the state is PPT).
        return is_ppt(rho, 2, dim_arr) and is_positive_semidefinite(rho)

    # In the 2-qubit case, an analytic formula is known for whether or not a state has a
    # (2-copy, non-PPT) symmetric extension that is much faster to use than semidefinite
    # programming [CJKLZB14]_.
    if level == 2 and not ppt and dim_x == 2 and dim_y == 2:
        return np.trace(np.linalg.matrix_power(partial_trace(rho, [0], [dim_x, dim_y]), 2)) >= np.trace(
            np.linalg.matrix_power(rho, 2)
        ) - 4 * np.sqrt(np.linalg.det(rho))

    # Otherwise, use semidefinite programming to find a symmetric extension.
    # We solve a feasibility SDP: find sigma on X ⊗ Y^⊗level such that
    # tr_{Y_2,...,Y_level}(sigma) = rho, sigma >= 0, sigma is symmetric
    # under permutations of Y copies, and (optionally) PPT constraints hold.
    dim_list = np.int_([dim_x] + [dim_y] * level)
    sys_list = list(range(2, 2 + level - 1))
    sym = symmetric_projection(dim_y, level)
    dim_total = int(np.prod(dim_list))

    sigma = cvxpy.Variable((dim_total, dim_total), hermitian=True)

    constraints = [
        partial_trace(sigma, sys_list, dim_list) == rho,
        sigma >> 0,
        np.kron(np.identity(dim_x), sym) @ sigma @ np.kron(np.identity(dim_x), sym) == sigma,
    ]

    if ppt:
        constraints.append(partial_transpose(sigma, [0], dim_list) >> 0)
        for sys in range(level - 1):
            constraints.append(partial_transpose(sigma, [sys + 2], dim_list) >> 0)

    problem = cvxpy.Problem(cvxpy.Minimize(0), constraints)
    problem.solve()

    return problem.status == "optimal"
