"""Calculate the Schmidt rank of a quantum state."""

import numpy as np

from toqito.perms import swap


def schmidt_rank(rho: np.ndarray, dim: int | list[int] | np.ndarray | None = None) -> int | float:
    r"""Compute the Schmidt rank :footcite:`WikiScmidtDecomp`.

    For complex Euclidean spaces :math:`\mathcal{X}` and :math:`\mathcal{Y}`, a pure state
    :math:`u \in \mathcal{X} \otimes \mathcal{Y}` possesses an expansion of the form:

    .. math::
        u = \sum_{i} \lambda_i v_i w_i,

    where :math:`v_i \in \mathcal{X}` and :math:`w_i \in \mathcal{Y}` are orthonormal states.

    The Schmidt coefficients are calculated from

    .. math::
        A = \text{Tr}_{\mathcal{B}}(u^* u).

    The Schmidt rank is the number of non-zero eigenvalues of :math:`A`. The Schmidt rank allows us
    to determine if a given state is entangled or separable. For instance:

        - If the Schmidt rank is 1: The state is separable,
        - If the Schmidt rank > 1: The state is entangled.

    Compute the Schmidt rank of the input :code:`rho`, provided as either a vector or a matrix that
    is assumed to live in bipartite space, where both subsystems have dimension equal to
    :code:`sqrt(len(vec))`.

    The dimension may be specified by the 1-by-2 vector :code:`dim` and the rank in that case is
    determined as the number of Schmidt coefficients larger than :code:`tol`.

    Examples
    ==========

    Computing the Schmidt rank of the entangled Bell state should yield a value greater than one.

    .. jupyter-execute::

        from toqito.states import bell
        from toqito.state_props import schmidt_rank
        rho = bell(0) @ bell(0).conj().T
        schmidt_rank(rho)


    Computing the Schmidt rank of the entangled singlet state should yield a value greater than
    :math:`1`.

    .. jupyter-execute::

        from toqito.states import bell
        from toqito.state_props import schmidt_rank
        u = bell(2) @ bell(2).conj().T
        schmidt_rank(u)


    Computing the Schmidt rank of a separable state should yield a value equal to :math:`1`.

    .. jupyter-execute::

        from toqito.states import basis
        from toqito.state_props import schmidt_rank
        import numpy as np
        e_0, e_1 = basis(2, 0), basis(2, 1)
        e_00 = np.kron(e_0, e_0)
        e_01 = np.kron(e_0, e_1)
        e_10 = np.kron(e_1, e_0)
        e_11 = np.kron(e_1, e_1)
        rho = 1 / 2 * (e_00 - e_01 - e_10 + e_11)
        rho = rho @ rho.conj().T
        schmidt_rank(rho)


    References
    ==========
    .. footbibliography::



    :param rho: A bipartite vector or matrix to have its Schmidt rank computed.
    :param dim: A 1-by-2 vector or matrix.
    :return: The Schmidt rank of :code:`rho`.

    """
    # If the input is provided as a matrix, compute the operator Schmidt rank.
    if len(rho.shape) == 2:
        if rho.shape[0] != 1 and rho.shape[1] != 1:
            return _operator_schmidt_rank(rho, dim)

    # Otherwise, compute the Schmidt rank for the vector.
    slv = int(np.round(np.sqrt(len(rho))))

    if dim is None:
        dim_arr = np.array([slv, len(rho) / slv], dtype=int)
    elif isinstance(dim, int):
        dim_arr = np.array([dim, len(rho) / dim], dtype=int)
        dim_arr[1] = np.round(dim_arr[1])
    elif isinstance(dim, list):
        dim_arr = np.array(dim)
    else:
        dim_arr = dim

    return np.linalg.matrix_rank(np.reshape(rho, dim_arr[::-1]))


def _operator_schmidt_rank(rho: np.ndarray, dim: int | list[int] | np.ndarray | None = None) -> int | float:
    """Operator Schmidt rank of variable.

    If the input is provided as a density operator instead of a vector, compute
    the operator Schmidt rank.
    """
    if dim is None:
        dim_x = rho.shape
        sqrt_dim = np.round(np.sqrt(dim_x))
        dim_arr = np.array([[sqrt_dim[0], sqrt_dim[0]], [sqrt_dim[1], sqrt_dim[1]]])
    elif isinstance(dim, list):
        dim_arr = np.array(dim)
    elif isinstance(dim, int):
        dim_arr = np.array([dim, len(rho) / dim], dtype=int)
        dim_arr[1] = np.round(dim_arr[1])
    else:
        dim_arr = dim

    if min(dim_arr.shape) == 1 or len(dim_arr.shape) == 1:
        dim_arr = np.array([dim_arr, dim_arr])

    op_1 = rho.reshape(int(np.prod(np.prod(dim_arr))), 1)
    swap_dim = np.concatenate((dim_arr[1, :].astype(int), dim_arr[0, :].astype(int)))
    op_2 = swap(op_1, [2, 3], swap_dim).reshape(-1, 1)

    return schmidt_rank(op_2, np.prod(dim_arr, axis=0).astype(int))
