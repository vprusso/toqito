"""Pure to mixed operation."""
import numpy as np


def pure_to_mixed(phi: np.ndarray) -> np.ndarray:
    r"""
    Convert a state vector or density matrix to a density matrix.

    Examples
    ==========

    It is possible to convert a pure state vector to a mixed state vector using the :code:`toqito`
    package. Consider the following Bell state

    .. math::
        u = \frac{1}{\sqrt{2}} \left( |00 \rangle + |11 \rangle \right).

    The corresponding mixed state from :math:`u` is calculated as

    .. math::
        \rho = u u^* = \frac{1}{2} \begin{pmatrix}
                                        1 & 0 & 0 & 1 \\
                                        0 & 0 & 0 & 0 \\
                                        0 & 0 & 0 & 0 \\
                                        1 & 0 & 0 & 1
                                   \end{pmatrix}

    Using :code:`toqito`, we can obtain this matrix as follows.

    >>> from toqito.states import bell
    >>> from toqito.state_ops import pure_to_mixed
    >>> phi = bell(0)
    >>> pure_to_mixed(phi)
    [[0.5, 0. , 0. , 0.5],
     [0. , 0. , 0. , 0. ],
     [0. , 0. , 0. , 0. ],
     [0.5, 0. , 0. , 0.5]]

    We can also give matrix inputs to the function in :code:`toqito`.

    >>> from toqito.states import bell
    >>> from toqito.state_ops import pure_to_mixed
    >>> phi = bell(0) * bell(0).conj().T
    >>> pure_to_mixed(phi)
    [[0.5, 0. , 0. , 0.5],
     [0. , 0. , 0. , 0. ],
     [0. , 0. , 0. , 0. ],
     [0.5, 0. , 0. , 0.5]])

    :param phi: A density matrix or a pure state vector.
    :return: density matrix representation of :code:`phi`, regardless of whether :code:`phi` is
             itself already a density matrix or if if is a pure state vector.
    """
    # Compute the size of `phi`. If it's already a mixed state, leave it alone.
    # If it's a vector (pure state), make it into a density matrix.
    row_dim, col_dim = phi.shape[0], phi.shape[1]

    # It's a pure state vector.
    if min(row_dim, col_dim) == 1:
        return phi * phi.conj().T
    # It's a density matrix.
    if row_dim == col_dim:
        return phi
    # It's neither.
    raise ValueError("InvalidDim: `phi` must be either a vector or square " "matrix.")
