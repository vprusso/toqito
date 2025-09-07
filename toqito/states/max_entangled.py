"""Maximally entangled states are states where the qubits are completely dependent on each other.

In these states, when a measurement is taken on one of the qubits, the state of the other qubits is automatically known.
"""

import numpy as np
from scipy.sparse import coo_array


def max_entangled(dim: int, is_sparse: bool = False, is_normalized: bool = True) -> [np.ndarray, coo_array]:
    r"""Produce a maximally entangled bipartite pure state :footcite:`WikiMaxEnt`.

    Produces a maximally entangled pure state as above that is sparse if :code:`is_sparse = True` and is full if
    :code:`is_sparse = False`. The pure state is normalized to have Euclidean norm 1 if :code:`is_normalized = True`,
    and it is unnormalized (i.e. each entry in the vector is 0 or 1 and the Euclidean norm of the vector is
    :code:`sqrt(dim)` if :code:`is_normalized = False`.

    Examples
    ==========

    We can generate the canonical :math:`2`-dimensional maximally entangled state

    .. math::
        u = \frac{1}{\sqrt{2}} \left( |00 \rangle + |11 \rangle \right)

    using :code:`|toqito⟩` as follows.

    .. jupyter-execute::

        from toqito.states import max_entangled
        max_entangled(2)


    By default, the state returned in normalized, however we can generate the unnormalized state

    .. math::
        v = |00\rangle + |11 \rangle

    using :code:`|toqito⟩` as follows.

    .. jupyter-execute::

        from toqito.states import max_entangled
        max_entangled(2, False, False)


    References
    ==========
    .. footbibliography::



    :param dim: Dimension of the entangled state.
    :param is_sparse: `True` if vector is sparse and `False` otherwise.
    :param is_normalized: `True` if vector is normalized and `False` otherwise.
    :return: The maximally entangled state of dimension :code:`dim`.

    """
    norm_factor = 1 / np.sqrt(dim) if is_normalized else 1.0
    idx = np.arange(dim) * (dim + 1)  # positions of nonzero entries in flattened form.

    if is_sparse:
        # Construct sparse vector directly.
        data = np.full(dim, norm_factor)
        psi = coo_array((data, (idx, np.zeros(dim))), shape=(dim**2, 1))
        return psi

    psi = np.zeros((dim**2, 1), dtype=float)
    psi[idx, 0] = norm_factor
    return psi
