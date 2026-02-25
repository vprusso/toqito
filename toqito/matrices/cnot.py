"""CNOT matrix generates the CNOT operator matrix."""

import numpy as np


def cnot() -> np.ndarray:
    r"""Produce the CNOT matrix [@WikiCNOT].

    The CNOT matrix is defined as

    \[
        \text{CNOT} =
        \begin{pmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & 0 & 1 \\
            0 & 0 & 1 & 0
        \end{pmatrix}.
    \]

    Examples:
    ```python exec="1" source="above"
    from toqito.matrices import cnot

    print(cnot())
    ```

    Returns:
        The CNOT matrix.

    """
    return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
