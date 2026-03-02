"""Calcultes the purity of a quantum state."""

import numpy as np

from toqito.matrix_props import is_density


def purity(rho: np.ndarray) -> float:
    r"""Compute the purity of a quantum state [@wikipediapurity].

    The negativity of a subsystem can be defined in terms of a density matrix \(\rho\): The
    purity of a quantum state \(\rho\) is defined as

    \[
        \text{Tr}(\rho^2),
    \]

    where \(\text{Tr}\) is the trace function.

    Examples:
        Consider the following scaled state defined as the scaled identity matrix

        \[
            \rho = \frac{1}{4} \begin{pmatrix}
                             1 & 0 & 0 & 0 \\
                             0 & 1 & 0 & 0 \\
                             0 & 0 & 1 & 0 \\
                             0 & 0 & 0 & 1
                           \end{pmatrix} \in \text{D}(\mathcal{X}).
        \]

        Calculating the purity of \(\rho\) yields \(\frac{1}{4}\). This can be observed using
        `|toqito⟩` as follows.

        ```python exec="1" source="above" session="purity_example"
        from toqito.state_props import purity
        import numpy as np
        print(purity(np.identity(4) / 4))
        ```


        Calculate the purity of the Werner state:

        ```python exec="1" source="above" session="purity_example"
        from toqito.states import werner
        rho = werner(2, 1 / 4)
        print(purity(rho))
        ```

    Raises:
        ValueError: If matrix is not density operator.

    Args:
        rho: A density matrix of a pure state vector.

    Returns:
        A value between 0 and 1 that corresponds to the purity of \(\rho\).

    """
    if not is_density(rho):
        raise ValueError("Purity is only defined for density operators.")
    # "np.real" get rid of the close-to-0 imaginary part.
    return np.real(np.trace(np.linalg.matrix_power(rho, 2)))
