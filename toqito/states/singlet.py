"""Generalized singlet state is a singlet state of `n` qubits in the invariant space of alternating representation."""

import numpy as np

from toqito.perms import swap_operator


def singlet(dim: int) -> np.ndarray:
    r"""Produce a generalized singlet state acting on two n-dimensional systems [@cabello2002nparticle].

    Examples:
        For \(n = 2\) this generates the following matrix

        \[
            S = \frac{1}{2} \begin{pmatrix}
                            0 & 0 & 0 & 0 \\
                            0 & 1 & -1 & 0 \\
                            0 & -1 & 1 & 0 \\
                            0 & 0 & 0 & 0
                        \end{pmatrix}
        \]

        which is equivalent to \(|\phi_s \rangle \langle \phi_s |\) where

        \[
            |\phi_s\rangle = \frac{1}{\sqrt{2}} \left( |01 \rangle - |10 \rangle \right)
        \]

        is the singlet state. This can be computed via `|toqito⟩` as follows:

        ```python exec="1" source="above"
        from toqito.states import singlet
        dim = 2
        print(singlet(dim))
        ```


        It is possible for us to consider higher dimensional singlet states. For instance, we can consider the
        \(3\)-dimensional Singlet state as follows:

        ```python exec="1" source="above"
        from toqito.states import singlet
        dim = 3
        print(singlet(dim))
        ```

    Args:
        dim: The dimension of the generalized singlet state.

    Returns:
        The singlet state of dimension `dim`.

    """
    return (np.identity(dim**2) - swap_operator([dim, dim])) / ((dim**2) - dim)
