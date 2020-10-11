"""Check if state is product vector."""
from typing import List, Union

import numpy as np

from toqito.state_ops import schmidt_decomposition


def is_product_vector(vec: np.ndarray, dim: Union[int, List[int]] = None) -> bool:
    r"""
    Determine if a given vector is a product vector [WikProdState]_.

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

    Calculating the rank of :math:`\rho` yields that the :math:`\rho` is a pure state. This can be
    confirmed in :code:`toqito` as follows:

    >>> from toqito.states import bell
    >>> from toqito.state_props import is_pure
    >>> u = bell(0)
    >>> rho = u * u.conj().T
    >>> is_pure(rho)
    True

    It is also possible to determine whether a set of density matrices are pure. For instance, we
    can see that the density matrices corresponding to the four Bell states yield a result of
    :code:`True` indicating that all states provided to the function are pure.

    >>> from toqito.states import bell
    >>> from toqito.state_props import is_pure
    >>> u0, u1, u2, u3 = bell(0), bell(1), bell(2), bell(3)
    >>> rho0 = u0 * u0.conj().T
    >>> rho1 = u1 * u1.conj().T
    >>> rho2 = u2 * u2.conj().T
    >>> rho3 = u3 * u3.conj().T
    >>>
    >>> is_pure([rho0, rho1, rho2, rho3])
    True

    References
    ==========
    .. [WikProdState] Wikipedia: Quantum state - Pure states
        https://en.wikipedia.org/wiki/Quantum_state#Pure_states

    :param vec: The vector to check.
    :param dim: The dimension of the vector
    :return: True if :code:`vec` is a product vector and False otherwise.
    """
    return _is_product_vector(vec, dim)[0][0]


def _is_product_vector(vec: np.ndarray, dim: Union[int, List[int]] = None) -> [int, bool]:
    """
    Determine if a given vector is a product vector recursive helper.

    :param vec: The vector to check.
    :param dim: The dimension of the vector
    :return: :code:`True` if :code:`vec` is a product vector and :code:`False`
             otherwise.
    """
    if dim is None:
        dim = np.round(np.sqrt(len(vec)))
    if isinstance(dim, list):
        dim = np.array(dim)

    # Allow the user to enter a single number for dim.
    if isinstance(dim, float):
        num_sys = 1
    else:
        num_sys = len(dim)

    if num_sys == 1:
        dim = np.array([dim, len(vec) // dim])
        dim[1] = np.round(dim[1])
        num_sys = 2

    dec = 0
    # If there are only two subsystems, just use the Schmidt decomposition.
    if num_sys == 2:
        singular_vals, u_mat, vt_mat = schmidt_decomposition(vec, dim, 2)
        ipv = singular_vals[1] <= np.prod(dim) * np.spacing(singular_vals[0])

        # Provide this even if not requested, since it is needed if this
        # function was called as part of its recursive algorithm (see below)
        if ipv:
            u_mat = u_mat * np.sqrt(singular_vals[0])
            vt_mat = vt_mat * np.sqrt(singular_vals[0])
            dec = [u_mat[:, 0], vt_mat[:, 0]]
    else:
        new_dim = [dim[0] * dim[1]]
        new_dim.extend(dim[2:])
        ipv, dec = _is_product_vector(vec, new_dim)
        if ipv:
            ipv, tdec = _is_product_vector(dec[0], [dim[0], dim[1]])
            if ipv:
                dec = [tdec, dec[1:]]

    return ipv, dec
