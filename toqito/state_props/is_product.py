"""Check if state is product."""


import numpy as np

from toqito.perms import permute_systems, swap
from toqito.state_ops import schmidt_decomposition


def is_product(rho: np.ndarray, dim: int | list[int] | np.ndarray = None) -> bool | np.ndarray:
    r"""Determine if a given vector is a product state :cite:`WikiSepSt`.

    If the input is deemed to be product, then the product decomposition is also
    returned.

    Examples
    ==========
    Consider the following Bell state

    .. math::
        u = \frac{1}{\sqrt{2}} \left( |00 \rangle + |11 \rangle \right) \in \mathcal{X}.

    The corresponding density matrix of :math:`u` may be calculated by:

    .. math::
        \rho = u u^* = \frac{1}{2} \begin{pmatrix}
                         1 & 0 & 0 & 1 \\
                         0 & 0 & 0 & 0 \\
                         0 & 0 & 0 & 0 \\
                         1 & 0 & 0 & 1
                       \end{pmatrix} \in \text{D}(\mathcal{X}).

    We can provide the input as either the vector :math:`u` or the denisty matrix :math:`\rho`.
    In either case, this represents an entangled state (and hence a non-product state).

    >>> from toqito.state_props import is_product
    >>> from toqito.states import bell
    >>> rho = bell(0) * bell(0).conj().T
    >>> u_vec = bell(0)
    >>> is_product(rho)
    (array([False]), None)
    >>>
    >>> is_product(u_vec)
    (array([False]), None)

    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    :param rho: The vector or matrix to check.
    :param dim: The dimension of the input.
    :return: :code:`True` if :code:`rho` is a product vector and :code:`False` otherwise.

    """
    return _is_product(rho, dim)


def _is_product(rho: np.ndarray, dim: int | list[int] = None) -> list[int, bool]:
    """Determine if input is a product state recursive helper.

    :param rho: The vector or matrix to check.
    :param dim: The dimension of the input.
    :return: :code:`True` if :code:`rho` is a product vector and :code:`False` otherwise.
    """
    # If the input is provided as a matrix, compute the operator Schmidt rank.
    if len(rho.shape) == 2:
        if rho.shape[0] != 1 and rho.shape[1] != 1:
            return _operator_is_product(rho, dim)

    if dim is None:
        dim = np.round(np.sqrt(len(rho)))
    if isinstance(dim, list):
        dim = np.array(dim)

    # Allow the user to enter a single number for dim.
    if isinstance(dim, float):
        num_sys = 1
    else:
        num_sys = len(dim)

    if num_sys == 1:
        dim = np.array([dim, len(rho) // dim])
        dim[1] = np.round(dim[1])
        num_sys = 2

    dec = None
    # If there are only two subsystems, just use the Schmidt decomposition.
    if num_sys == 2:
        singular_vals, u_mat, vt_mat = schmidt_decomposition(rho, dim, 2)

        # Provide this even if not requested, since it is needed if this
        # function was called as part of its recursive algorithm (see below)
        if ipv := singular_vals[1] <= np.prod(dim) * np.spacing(singular_vals[0]):
            u_mat = u_mat * np.sqrt(singular_vals[0])
            vt_mat = vt_mat * np.sqrt(singular_vals[0])
            dec = [u_mat[:, 0], vt_mat[:, 0]]
    else:
        new_dim = [dim[0] * dim[1]]
        new_dim.extend(dim[2:])
        ipv, dec = _is_product(rho, new_dim)
        if ipv:
            ipv, tdec = _is_product(dec[0], [dim[0], dim[1]])
            if ipv:
                dec = [*tdec, *dec[1:]]
    return ipv, dec


def _operator_is_product(rho: np.ndarray, dim: int | list[int] = None) -> list[int, bool]:
    r"""Determine if a given matrix is a product operator.

    Given an input `rho` provided as a matrix, determine if it is a product
    state.
    :param rho: The matrix to check.
    :param dim: The dimension of the matrix
    :return: :code:`True` if :code:`rho` is product and :code:`False` otherwise.
    """
    if dim is None:
        dim_x = rho.shape
        sqrt_dim = np.round(np.sqrt(dim_x))

        dim = np.array([[sqrt_dim[0], sqrt_dim[0]], [sqrt_dim[1], sqrt_dim[1]]])

    if isinstance(dim, list):
        dim = np.array(dim)

    num_sys = len(dim)

    # Allow the user to enter a vector for `dim` if `rho` is square.
    if min(dim.shape) == 1 or len(dim.shape) == 1:
        dim = dim.T.flatten()
        dim = np.array([dim, dim])

    op_1 = rho.reshape(int(np.prod(np.prod(dim))), 1)
    perm = swap(np.array(list(range(2 * num_sys))), [1, 2], [2, num_sys]) + 1
    perm_dim = np.concatenate((dim[1, :].astype(int), dim[0, :].astype(int)))
    op_3 = permute_systems(op_1, perm, perm_dim).reshape(-1, 1)

    return is_product(op_3, np.prod(dim, axis=0).astype(int))
