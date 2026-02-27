"""Isotropic state is a bipartite quantum state.

These states are separable for α ≤ 1/(d+1), but are otherwise entangled.
"""

import numpy as np

from toqito.states import max_entangled


def isotropic(dim: int, alpha: float) -> np.ndarray:
    r"""Produce a isotropic state [@Horodecki_1998_Reduction].

    Returns the isotropic state with parameter `alpha` acting on (`dim`-by-`dim`)-dimensional space.
    The isotropic state has the following form

    \[
        \begin{equation}
            \rho_{\alpha} = \frac{1 - \alpha}{d^2} \mathbb{I} \otimes
            \mathbb{I} + \alpha |\psi_+ \rangle \langle \psi_+ | \in
            \mathbb{C}^d \otimes \mathbb{C}^2
        \end{equation}
    \]

    where \(|\psi_+ \rangle = \frac{1}{\sqrt{d}} \sum_j |j \rangle \otimes |j \rangle\) is the maximally entangled
    state.

    Examples:
        To generate the isotropic state with parameter \(\alpha=1/2\), we can make the following call to
        `|toqito⟩` as

        ```python exec="1" source="above"
        from toqito.states import isotropic
        print(isotropic(3, 1 / 2))
        ```

    Args:
        dim: The local dimension.
        alpha: The parameter of the isotropic state.

    Returns:
        Isotropic state of dimension `dim`.

    """
    psi = max_entangled(dim, False, False)
    return (1 - alpha) * np.identity(dim**2) / dim**2 + alpha * psi @ psi.conj().T / dim
