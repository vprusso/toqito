"""Schmidt rank of state."""


import numpy as np

from toqito.perms import swap


def schmidt_rank(rho: np.ndarray, dim: int | list[int] | np.ndarray = None) -> float:
    r"""Compute the Schmidt rank :cite:`WikiScmidtDecomp`.

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

    >>> from toqito.states import bell
    >>> from toqito.state_props import schmidt_rank
    >>> rho = bell(0) @ bell(0).conj().T
    >>> schmidt_rank(rho)
    4

    Computing the Schmidt rank of the entangled singlet state should yield a value greater than
    :math:`1`.

    >>> from toqito.states import bell
    >>> from toqito.state_props import schmidt_rank
    >>> u = bell(2) @ bell(2).conj().T
    >>> schmidt_rank(u)
    4

    Computing the Schmidt rank of a separable state should yield a value equal to :math:`1`.

    >>> from toqito.states import basis
    >>> from toqito.state_props import schmidt_rank
    >>> import numpy as np
    >>> e_0, e_1 = basis(2, 0), basis(2, 1)
    >>> e_00 = np.kron(e_0, e_0)
    >>> e_01 = np.kron(e_0, e_1)
    >>> e_10 = np.kron(e_1, e_0)
    >>> e_11 = np.kron(e_1, e_1)
    >>>
    >>> rho = 1 / 2 * (e_00 - e_01 - e_10 + e_11)
    >>> rho = rho @ rho.conj().T
    >>> schmidt_rank(rho)
    1

    References
    ==========
    .. bibliography::
        :filter: docname in docnames


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
        dim = slv
    if isinstance(dim, int):
        dim = np.array([dim, len(rho) / dim], dtype=int)  # pylint: disable=redefined-variable-type
        dim[1] = np.round(dim[1])

    return np.linalg.matrix_rank(np.reshape(rho, dim[::-1]))


def _operator_schmidt_rank(rho: np.ndarray, dim: int | list[int] | np.ndarray = None) -> float:
    """Operator Schmidt rank of variable.

    If the input is provided as a density operator instead of a vector, compute
    the operator Schmidt rank.
    """
    if dim is None:
        dim_x = rho.shape
        sqrt_dim = np.round(np.sqrt(dim_x))
        dim = np.array([[sqrt_dim[0], sqrt_dim[0]], [sqrt_dim[1], sqrt_dim[1]]])

    if isinstance(dim, list):
        dim = np.array(dim)

    if isinstance(dim, int):
        dim = np.array([dim, len(rho) / dim], dtype=int)
        dim[1] = np.round(dim[1])

    if min(dim.shape) == 1 or len(dim.shape) == 1:
        dim = np.array([dim, dim])

    op_1 = rho.reshape(int(np.prod(np.prod(dim))), 1)
    swap_dim = np.concatenate((dim[1, :].astype(int), dim[0, :].astype(int)))
    op_2 = swap(op_1, [2, 3], swap_dim).reshape(-1, 1)

    return schmidt_rank(op_2, np.prod(dim, axis=0).astype(int))
