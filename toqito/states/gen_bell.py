"""Generalized Bell state represents a bigger set of Bell states.

This set includes the standard bell states and other higher dimensional bell states as well.
Generalized Bell states are the basis of multidimensional bipartite states having maximum entanglement.
"""

import numpy as np

from toqito.matrices import gen_pauli
from toqito.perms import vec


def gen_bell(k_1: int, k_2: int, dim: int) -> np.ndarray:
    r"""Produce a generalized Bell state [@sych2009complete].

    Produces a generalized Bell state. Note that the standard Bell states can be recovered as:

        bell(0) : gen_bell(0, 0, 2)

        bell(1) : gen_bell(0, 1, 2)

        bell(2) : gen_bell(1, 0, 2)

        bell(3) : gen_bell(1, 1, 2)


    Examples:
        For \(d = 2\) and \(k_1 = k_2 = 0\), this generates the following matrix

        \[
            G = \frac{1}{2} \begin{pmatrix}
                            1 & 0 & 0 & 1 \\
                            0 & 0 & 0 & 0 \\
                            0 & 0 & 0 & 0 \\
                            1 & 0 & 0 & 1
                        \end{pmatrix}
        \]

        which is equivalent to \(|\phi_0 \rangle \langle \phi_0 |\) where

        \[
            |\phi_0\rangle = \frac{1}{\sqrt{2}} \left( |00 \rangle + |11 \rangle \right)
        \]

        is one of the four standard Bell states. This can be computed via `|toqito⟩` as follows.

        ```python exec="1" source="above"
        from toqito.states import gen_bell
        dim = 2
        k_1 = 0
        k_2 = 0
        print(gen_bell(k_1, k_2, dim))
        ```

        It is possible for us to consider higher dimensional Bell states. For instance, we can consider
        the \(3\)-dimensional Bell state for \(k_1 = k_2 = 0\) as follows.

        ```python exec="1" source="above"
        from toqito.states import gen_bell
        dim = 3
        k_1 = 0
        k_2 = 0
        print(gen_bell(k_1, k_2, dim))
        ```

    Args:
        k_1: An integer 0 <= k_1 <= n.
        k_2: An integer 0 <= k_2 <= n.
        dim: The dimension of the generalized Bell state.

    """
    gen_pauli_w = gen_pauli(k_1, k_2, dim)
    return 1 / dim * vec(gen_pauli_w) @ vec(gen_pauli_w).conj().T
