"""Helstrom-Holevo metric gives the bst success probability to distinguish two mixed states."""

import numpy as np

from toqito.matrix_props import is_density, trace_norm


def helstrom_holevo(rho: np.ndarray, sigma: np.ndarray) -> float | np.floating:
    r"""Compute the Helstrom-Holevo distance between density matrices [@wikipediaholevo].

    In general, the best success probability to discriminate two mixed states represented by
    \(\rho\) and \(\sigma\) is given by [@wikipediaholevo].

    \[
        \frac{1}{2}+\frac{1}{2} \left(\frac{1}{2} \left|\rho - \sigma \right|_1\right).
    \]

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

        Calculating the Helstrom-Holevo distance of states that are identical yield a value of
        \(1/2\). This can be verified in `|toqito⟩` as follows.

        ```python exec="1" source="above"
        import numpy as np
        from toqito.states import basis
        from toqito.state_metrics import helstrom_holevo

        e_0, e_1 = basis(2, 0), basis(2, 1)
        e_00 = np.kron(e_0, e_0)
        e_11 = np.kron(e_1, e_1)

        u_vec = 1 / np.sqrt(2) * (e_00 + e_11)
        rho = u_vec @ u_vec.conj().T
        sigma = rho

        print(helstrom_holevo(rho, sigma))
        ```

    Raises:
        ValueError: If matrices are not density operators.

    Args:
        rho: Density operator.
        sigma: Density operator.

    Returns:
        The Helstrom-Holevo distance between `rho` and `sigma`.

    """
    if not is_density(rho) or not is_density(sigma):
        raise ValueError("Helstrom-Holevo is only defined for density operators.")
    return 1 / 2 + 1 / 2 * (trace_norm(rho - sigma)) / 2
