"""Computes the trace norm metric of a density matrix."""

import numpy as np


def trace_norm(rho: np.ndarray) -> float | np.floating:
    r"""Compute the trace norm of the state [@Quantiki_TrNorm].

    Also computes the operator 1-norm when inputting an operator.

    The trace norm \(||\rho||_1\) of a density matrix \(\rho\) is the sum of the singular
    values of \(\rho\). The singular values are the roots of the eigenvalues of
    \(\rho \rho^*\).

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

    It can be observed using `|toqito⟩` that \(||\rho||_1 = 1\) as follows.

    ```python exec="1" source="above"
    from toqito.states import bell
    from toqito.matrix_props import trace_norm

    rho = bell(0) @ bell(0).conj().T

    print(trace_norm(rho))
    ```

    Args:
        rho: Density operator.

    Returns:
        The trace norm of `rho`.

    """
    return np.linalg.norm(rho, ord="nuc")
