"""BB84 states represent the BB84 basis states, which are based on BB84, a quantum key distribution scheme.

In the BB884 scheme, each qubit is encoded with one of the 4 polarization states: 0, 1, +45° or -45°.
"""

import numpy as np

from toqito.matrices import standard_basis


def bb84() -> list[list[np.ndarray]]:
    r"""Obtain the BB84 basis states [@wikipediabb84].

    The BB84 basis states are defined as

    \[
        |0\rangle := \begin{pmatrix} 1 \\ 0 \end{pmatrix}, \quad \\
        |1\rangle := \begin{pmatrix} 0 \\ 1 \end{pmatrix}, \quad \\
        |+\rangle := \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ 1 \end{pmatrix}, \quad \\
        |-\rangle := \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ -1 \end{pmatrix}.
    \]

    Examples:
        The BB84 basis states can be obtained in `|toqito⟩` as follows in the form of a list of
        arrays.

        ```python exec="1" source="above"
        from toqito.states import bb84
        print(bb84())
        ```

    Returns:
        The four BB84 basis states.

    """
    # Computational basis states |0>, |1>:
    e_0, e_1 = standard_basis(2)
    # Plus/minus basis |+>, |->
    e_p, e_m = (e_0 + e_1) / np.sqrt(2), (e_0 - e_1) / np.sqrt(2)
    return [[e_0, e_1], [e_p, e_m]]
