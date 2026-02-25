"""Checks if a quantum state is mixed."""

import numpy as np

from toqito.state_props import is_pure


def is_mixed(state: np.ndarray) -> bool:
    r"""Determine if a given quantum state is mixed [@WikiMixedSt].

    A mixed state by definition is a state that is not pure.

    Examples:

    Consider the following density matrix:

    \[
        \rho =  \begin{pmatrix}
                    \frac{3}{4} & 0 \\
                    0 & \frac{1}{4}
                \end{pmatrix} \in \text{D}(\mathcal{X}).
    \]

    Calculating the rank of \(\rho\) yields that the \(\rho\) is a mixed state. This can be
    confirmed in `|toqito⟩` as follows:

    ```python exec="1" source="above"
    from toqito.states import basis
    from toqito.state_props import is_mixed
    e_0, e_1 = basis(2, 0), basis(2, 1)
    rho = 3 / 4 * e_0 @ e_0.conj().T + 1 / 4 * e_1 @ e_1.conj().T
    print(is_mixed(rho))
    ```

    Args:
        state: The density matrix representing the quantum state.

    Returns:
        `True` if state is mixed and `False` otherwise.

    """
    return not is_pure(state)
