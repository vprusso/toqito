"""Generalized Bell state."""
import numpy as np

from toqito.matrices import gen_pauli
from toqito.matrix_ops import vec


def gen_bell(k_1: int, k_2: int, dim: int) -> np.ndarray:
    r"""
    Produce a generalized Bell state [DL09]_.

    Produces a generalized Bell state. Note that the standard Bell states can be recovered as:

    ```
    - `bell(0)` : `gen_bell(0, 0, 2)`
    - `bell(1)` : `gen_bell(0, 1, 2)`
    - `bell(2)` : `gen_bell(1, 0, 2)`
    - `bell(3)` : `gen_bell(1, 1, 2)`
    ```

    Examples
    ==========

    For :math:`d = 2` and :math:`k_1 = k_2 = 0`, this generates the following matrix

    .. math::
        G = \frac{1}{2} \begin{pmatrix}
                        1 & 0 & 0 & 1 \\
                        0 & 0 & 0 & 0 \\
                        0 & 0 & 0 & 0 \\
                        1 & 0 & 0 & 1
                    \end{pmatrix}

    which is equivalent to :math:`|\phi_0 \rangle \langle \phi_0 |` where

    .. math::
        |\phi_0\rangle = \frac{1}{\sqrt{2}} \left( |00 \rangle + |11 \rangle \right)

    is one of the four standard Bell states. This can be computed via :code:`toqito` as follows.

    >>> from toqito.states import gen_bell
    >>> dim = 2
    >>> k_1 = 0
    >>> k_2 = 0
    >>> gen_bell(k_1, k_2, dim)
    [[0.5+0.j, 0. +0.j, 0. +0.j, 0.5+0.j],
     [0. +0.j, 0. +0.j, 0. +0.j, 0. +0.j],
     [0. +0.j, 0. +0.j, 0. +0.j, 0. +0.j],
     [0.5+0.j, 0. +0.j, 0. +0.j, 0.5+0.j]]

    It is possible for us to consider higher dimensional Bell states. For instance, we can consider
    the :math:`3`-dimensional Bell state for :math:`k_1 = k_2 = 0` as follows.

    >>> from toqito.states import gen_bell
    >>> dim = 3
    >>> k_1 = 0
    >>> k_2 = 0
    >>> gen_bell(k_1, k_2, dim)
    [[0.33333333+0.j, 0.        +0.j, 0.        +0.j, 0.        +0.j,
      0.33333333+0.j, 0.        +0.j, 0.        +0.j, 0.        +0.j,
      0.33333333+0.j],
     [0.        +0.j, 0.        +0.j, 0.        +0.j, 0.        +0.j,
      0.        +0.j, 0.        +0.j, 0.        +0.j, 0.        +0.j,
      0.        +0.j],
     [0.        +0.j, 0.        +0.j, 0.        +0.j, 0.        +0.j,
      0.        +0.j, 0.        +0.j, 0.        +0.j, 0.        +0.j,
      0.        +0.j],
     [0.        +0.j, 0.        +0.j, 0.        +0.j, 0.        +0.j,
      0.        +0.j, 0.        +0.j, 0.        +0.j, 0.        +0.j,
      0.        +0.j],
     [0.33333333+0.j, 0.        +0.j, 0.        +0.j, 0.        +0.j,
      0.33333333+0.j, 0.        +0.j, 0.        +0.j, 0.        +0.j,
      0.33333333+0.j],
     [0.        +0.j, 0.        +0.j, 0.        +0.j, 0.        +0.j,
      0.        +0.j, 0.        +0.j, 0.        +0.j, 0.        +0.j,
      0.        +0.j],
     [0.        +0.j, 0.        +0.j, 0.        +0.j, 0.        +0.j,
      0.        +0.j, 0.        +0.j, 0.        +0.j, 0.        +0.j,
      0.        +0.j],
     [0.        +0.j, 0.        +0.j, 0.        +0.j, 0.        +0.j,
      0.        +0.j, 0.        +0.j, 0.        +0.j, 0.        +0.j,
      0.        +0.j],
     [0.33333333+0.j, 0.        +0.j, 0.        +0.j, 0.        +0.j,
      0.33333333+0.j, 0.        +0.j, 0.        +0.j, 0.        +0.j,
      0.33333333+0.j]]

    References
    ==========
    .. [DL09] Sych, Denis, and Gerd Leuchs.
        "A complete basis of generalized Bell states."
        New Journal of Physics 11.1 (2009): 013006.

    :param k_1: An integer 0 <= k_1 <= n.
    :param k_2: An integer 0 <= k_2 <= n.
    :param dim: The dimension of the generalized Bell state.
    """
    gen_pauli_w = gen_pauli(k_1, k_2, dim)
    return 1 / dim * vec(gen_pauli_w) * vec(gen_pauli_w).conj().T
