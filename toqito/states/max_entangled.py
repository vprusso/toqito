"""Maximally entangled state."""
import numpy as np

from scipy import sparse

from toqito.matrices import iden


def max_entangled(
    dim: int, is_sparse: bool = False, is_normalized: bool = True
) -> [np.ndarray, sparse.dia.dia_matrix]:
    r"""
    Produce a maximally entangled bipartite pure state [WikEnt]_.

    Produces a maximally entangled pure state as above that is sparse if :code:`is_sparse = True`
    and is full if :code:`is_sparse = False`. The pure state is normalized to have Euclidean norm 1
    if :code:`is_normalized = True`, and it is unnormalized (i.e. each entry in the vector is 0 or 1
    and the Euclidean norm of the vector is :code:`sqrt(dim)` if :code:`is_normalized = False`.

    Examples
    ==========

    We can generate the canonical :math:`2`-dimensional maximally entangled state

    .. math::
        u = \frac{1}{\sqrt{2}} \left( |00 \rangle + |11 \rangle \right)

    using :code:`toqito` as follows.

    >>> from toqito.states import max_entangled
    >>> max_entangled(2)
    [[0.70710678],
     [0.        ],
     [0.        ],
     [0.70710678]]

    By default, the state returned in normalized, however we can generate the unnormalized state

    .. math::
        v = |00\rangle + |11 \rangle

    using :code:`toqito` as follows.

    >>> from toqito.states import max_entangled
    >>> max_entangled(2, False, False)
    [[1.],
     [0.],
     [0.],
     [1.]]

    References
    ==========
    .. [WikEnt] Wikipedia: Quantum entanglement
        https://en.wikipedia.org/wiki/Quantum_entanglement

    :param dim: Dimension of the entangled state.
    :param is_sparse: `True` if vector is spare and `False` otherwise.
    :param is_normalized: `True` if vector is normalized and `False` otherwise.
    :return: The maximally entangled state of dimension :code:`dim`.
    """
    psi = np.reshape(iden(dim, is_sparse), (dim ** 2, 1))
    if is_normalized:
        psi = psi / np.sqrt(dim)
    return psi
