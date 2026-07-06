"""Breuer states represent the Breuer bound entangled states.

These states are based on the Breuer-Hall criterion.
"""

import numpy as np

from toqito.perms import symmetric_projection
from toqito.states import max_entangled


def breuer(dim: int, lam: float) -> np.ndarray:
    r"""Produce a Breuer state [@breuer2006optimal].

    Gives a Breuer state for two qudits of local dimension `dim`, with the `lam` parameter describing the
    weight of the singlet component as described in [@breuer2006optimal]. For even local dimensions
    \(d \geq 4\) and \(\lambda > 0\), this construction gives bound entangled states.

    This function was adapted from the QETLAB package [@qetlablink].

    Args:
        dim: Dimension of the Breuer state.
        lam: The weight of the singlet component.

    Returns:
        Breuer state of dimension `dim` with weight `lam`.

    Raises:
        ValueError: Dimension must be a positive even integer.
        ValueError: `lam` must be between 0 and 1.

    Examples:
        We can generate a Breuer bound entangled state of local dimension \(4\) with weight \(0.1\). For even
        local dimension at least \(4\) and any positive weight, the state satisfies the PPT criterion, but it
        is entangled.

        ```python exec="1" source="above" result="text"
        from toqito.states import breuer
        print(breuer(4, 0.1))
        ```

    """
    if dim % 2 == 1 or dim <= 0:
        raise ValueError(f"The value {dim} must be an even positive integer.")

    if not 0 <= lam <= 1:
        raise ValueError("The weight `lam` must be between 0 and 1.")

    v_mat = np.fliplr(np.diag((-1) ** np.mod(np.arange(1, dim + 1), 2)))
    max_entangled(dim)
    psi = np.dot(np.kron(np.identity(dim), v_mat), max_entangled(dim))

    return lam * (psi * psi.conj().T) + (1 - lam) * 2 * symmetric_projection(dim) / (dim * (dim + 1))
