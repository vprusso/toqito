"""Breuer states represent the Breuer bound entangled states.

These states are based on the Breuer-Hall criterion.
"""

import numpy as np

from toqito.perms import symmetric_projection
from toqito.states import max_entangled


def breuer(dim: int, lam: float) -> np.ndarray:
    r"""Produce a Breuer state [@Breuer_2006_Optimal].

    Gives a Breuer bound entangled state for two qudits of local dimension `dim`, with the
    `lam` parameter describing the weight of the singlet component as described in
    [@Breuer_2006_Optimal].

    This function was adapted from the QETLAB package.

    Examples:
    We can generate a Breuer state of dimension \(4\) with weight \(0.1\). For any weight above \(0\), the
    state will be bound entangled, that is, it will satisfy the PPT criterion, but it will be entangled.

    ```python exec="1" source="above"
    from toqito.states import breuer
    print(breuer(2, 0.1))
    ```

    Raises:
        ValueError: Dimension must be greater than or equal to 1.

    Args:
        dim: Dimension of the Breuer state.
        lam: The weight of the singlet component.

    Returns:
        Breuer state of dimension `dim` with weight `lam`.

    """
    if dim % 2 == 1 or dim <= 0:
        raise ValueError(f"The value {dim} must be an even positive integer.")

    v_mat = np.fliplr(np.diag((-1) ** np.mod(np.arange(1, dim + 1), 2)))
    max_entangled(dim)
    psi = np.dot(np.kron(np.identity(dim), v_mat), max_entangled(dim))

    return lam * (psi * psi.conj().T) + (1 - lam) * 2 * symmetric_projection(dim) / (dim * (dim + 1))
