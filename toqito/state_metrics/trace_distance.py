"""Trace distance metric gives a measure of distinguishability between two quantum states.

The trace distance is calculated via density matrices.
"""

import numpy as np

from toqito.matrix_props import is_density, trace_norm


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float | np.floating:
    r"""Compute the trace distance between density operators `rho` and `sigma`.

    The trace distance between \(\rho\) and \(\sigma\) is defined as

    \[
        \delta(\rho, \sigma) = \frac{1}{2} \left( \text{Tr}(\left| \rho - \sigma
         \right| \right).
    \]

    More information on the trace distance can be found in [@Quantiki_TrDist].

    Examples:
        Consider the following Bell state

        \[
            u = \frac{1}{\sqrt{2}} \left( |00 \rangle + |11 \rangle \right) \in \mathcal{X}.
        \]

        The corresponding density matrix of \(u\) may be calculated by:

        \[
            \rho = u u^* = \begin{pmatrix}
                             1 & 0 & 0 & 1 \\
                             0 & 0 & 0 & 0 \\
                             0 & 0 & 0 & 0 \\
                             1 & 0 & 0 & 1
                           \end{pmatrix} \in \text{D}(\mathcal{X}).
        \]

        The trace distance between \(\rho\) and another state \(\sigma\) is equal to \(0\) if any only if
        \(\rho = \sigma\). We can check this using the `|toqito⟩` package.

        ```python exec="1" source="above"
        from toqito.states import bell
        from toqito.state_metrics import trace_distance

        rho = bell(0) @ bell(0).conj().T
        sigma = rho

        print(trace_distance(rho, sigma))
        ```

    Raises:
        ValueError: If matrices are not of density operators.

    Args:
        rho: An input matrix.
        sigma: An input matrix.

    Returns:
        The trace distance between `rho` and `sigma`.

    """
    if not is_density(rho) or not is_density(sigma):
        raise ValueError("Trace distance only defined for density matrices.")
    return trace_norm(np.abs(rho - sigma)) / 2
