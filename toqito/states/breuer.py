"""Breuer state."""
import numpy as np

from toqito.perms import symmetric_projection
from toqito.states import max_entangled


def breuer(dim: int, lam: float) -> np.ndarray:
    r"""
    Produce a Breuer state [HPBreuer]_.

    Gives a Breuer bound entangled state for two qudits of local dimension :code:`dim`, with the
    :code:`lam` parameter describing the weight of the singlet component as described in [HPBreuer].

    This function was adapted from the QETLAB package.

    Examples
    ==========

    We can generate a Breuer state of dimension :math:`4` with weight :math:`0.1`. For any weight
    above :math:`0`, the state will be bound entangled, that is, it will satisfy the PPT criterion,
    but it will be entangled.

    >>> from toqito.states import breuer
    >>> breuer(2, 0.1)
    [[ 0.3  0.  -0.   0. ]
     [ 0.   0.2  0.1  0. ]
     [-0.   0.1  0.2 -0. ]
     [ 0.   0.  -0.   0.3]]

    References
    ==========
    .. [HPBreuer] H-P. Breuer. Optimal entanglement criterion for mixed quantum states.
       E-print: arXiv:quant-ph/0605036, 2006.

    :param dim: Dimension of the Breuer state.
    :param lam: The weight of the singlet component.
    :return: Breuer state of dimension :code:`dim` with weight :code:`lam`.
    """
    if dim % 2 == 1 or dim <= 0:
        raise ValueError(f"The value {dim} must be an even positive integer.")

    v_mat = np.fliplr(np.diag((-1) ** np.mod(np.arange(1, dim + 1), 2)))
    max_entangled(dim)
    psi = np.dot(np.kron(np.identity(dim), v_mat), max_entangled(dim))

    return lam * (psi * psi.conj().T) + (1 - lam) * 2 * symmetric_projection(dim) / (
        dim * (dim + 1)
    )
