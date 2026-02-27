"""Bell states represent the simplest examples of quantum entanglement of two qubits.

Also known as EPR pairs, Bell states comprise of four quantum states in a superposition of 0 and 1.
"""

import numpy as np


def bell(idx: int) -> np.ndarray:
    r"""Produce a Bell state [@WikiBellSt].

    Returns one of the following four Bell states depending on the value of `idx`:

    \[
        \begin{equation}
            \begin{aligned}
                u_0 = \frac{1}{\sqrt{2}} \left( |00 \rangle + |11 \rangle \right), &
                \qquad &
                u_1 = \frac{1}{\sqrt{2}} \left( |00 \rangle - |11 \rangle \right), \\
                u_2 = \frac{1}{\sqrt{2}} \left( |01 \rangle + |10 \rangle \right), &
                \qquad &
                u_3 = \frac{1}{\sqrt{2}} \left( |01 \rangle - |10 \rangle \right).
            \end{aligned}
        \end{equation}
    \]

    Examples:
        When `idx = 0`, this produces the following Bell state:

        \[
            u_0 = \frac{1}{\sqrt{2}} \left( |00 \rangle + |11 \rangle \right).
        \]

        Using `|toqito⟩`, we can see that this yields the proper state.

        ```python exec="1" source="above"
        from toqito.states import bell
        import numpy as np
        print(bell(0))
        ```

    Raises:
        ValueError: If `idx` is not an integer.

    Args:
        idx: A parameter in [0, 1, 2, 3]

    Returns:
        Bell state with index `idx`.

    """
    match idx:
        case 0:
            return 1 / np.sqrt(2) * np.array([[1], [0], [0], [1]])
        case 1:
            return 1 / np.sqrt(2) * np.array([[1], [0], [0], [-1]])
        case 2:
            return 1 / np.sqrt(2) * np.array([[0], [1], [1], [0]])
        case 3:
            return 1 / np.sqrt(2) * np.array([[0], [1], [-1], [0]])
    raise ValueError("Invalid integer value for Bell state.")
