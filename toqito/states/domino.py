"""Produce a domino state."""

import numpy as np

from toqito.states import basis


def domino(idx: int) -> np.ndarray:
    r"""Produce a domino state [@Bennett_1999_QuantumNonlocality][@Bennett_1999_UPB].

    The orthonormal product basis of domino states is given as

    \[
        \begin{equation}
            \begin{aligned}
            |\phi_0\rangle = |1\rangle |1 \rangle,
            \qquad
            |\phi_1\rangle = |0 \rangle \left(\frac{|0 \rangle + |1 \rangle}{\sqrt{2}} \right),
            & \qquad
            |\phi_2\rangle = |0\rangle \left(\frac{|0\rangle - |1\rangle}{\sqrt{2}}\right), \\
            |\phi_3\rangle = |2\rangle \left(\frac{|0\rangle + |1\rangle}{\sqrt{2}}\right), \qquad
            |\phi_4\rangle = |2\rangle \left(\frac{|0\rangle - |1\rangle}{\sqrt{2}}\right), & \qquad
            |\phi_5\rangle = \left(\frac{|0\rangle + |1\rangle}{\sqrt{2}}\right) |0\rangle, \\
            |\phi_6\rangle = \left(\frac{|0\rangle - |1\rangle}{\sqrt{2}}\right) |0\rangle, \qquad
            |\phi_7\rangle = \left(\frac{|0\rangle + |1\rangle}{\sqrt{2}}\right) |2\rangle, & \qquad
            |\phi_8\rangle = \left(\frac{|0\rangle - |1\rangle}{\sqrt{2}}\right) |2\rangle.
            \end{aligned}
        \end{equation}
    \]

    Returns one of the following nine domino states depending on the value of `idx`.

    Examples:
    When `idx = 0`, this produces the following Domino state

    \[
        |\phi_0 \rangle = |11 \rangle |11 \rangle.
    \]

    Using `|toqito⟩`, we can see that this yields the proper state.

    ```python exec="1" source="above"
    from toqito.states import domino
    print(domino(0))
    ```


    When `idx = 3`, this produces the following Domino state

    \[
        |\phi_3\rangle = |2\rangle \left(\frac{|0\rangle + |1\rangle}
        {\sqrt{2}}\right)
    \]

    Using `|toqito⟩`, we can see that this yields the proper state.

    ```python exec="1" source="above"
    from toqito.states import domino
    print(domino(3))
    ```

    Raises:
        ValueError: Invalid value for `idx`.

    Args:
        idx: A parameter in [0, 1, 2, 3, 4, 5, 6, 7, 8]

    Returns:
        Domino state of index `idx`.

    """
    e_0, e_1, e_2 = basis(3, 0), basis(3, 1), basis(3, 2)
    match idx:
        case 0:
            return np.kron(e_1, e_1)
        case 1:
            return np.kron(e_0, 1 / np.sqrt(2) * (e_0 + e_1))
        case 2:
            return np.kron(e_0, 1 / np.sqrt(2) * (e_0 - e_1))
        case 3:
            return np.kron(e_2, 1 / np.sqrt(2) * (e_1 + e_2))
        case 4:
            return np.kron(e_2, 1 / np.sqrt(2) * (e_1 - e_2))
        case 5:
            return np.kron(1 / np.sqrt(2) * (e_1 + e_2), e_0)
        case 6:
            return np.kron(1 / np.sqrt(2) * (e_1 - e_2), e_0)
        case 7:
            return np.kron(1 / np.sqrt(2) * (e_0 + e_1), e_2)
        case 8:
            return np.kron(1 / np.sqrt(2) * (e_0 - e_1), e_2)
    raise ValueError("Invalid integer value for Domino state.")
