"""Determines if an input is a quantum channel."""

import numpy as np

from toqito.channel_ops import kraus_to_choi
from toqito.channel_props import is_completely_positive, is_trace_preserving


def is_quantum_channel(
    phi: np.ndarray | list[list[np.ndarray]],
    rtol: float = 1e-05,
    atol: float = 1e-08,
) -> bool:
    r"""Determine whether the given input is a quantum channel.

    For more info, see Section 2.2.1: Definitions and Basic Notions Concerning Channels from
    [@Watrous_2018_TQI].

    A map \(\Phi \in \text{T} \left(\mathcal{X}, \mathcal{Y} \right)\) is a *quantum
    channel* for some choice of complex Euclidean spaces \(\mathcal{X}\)
    and \(\mathcal{Y}\), if it holds that:

    1. \(\Phi\) is completely positive.
    2. \(\Phi\) is trace preserving.

    Examples:
        We can specify the input as a list of Kraus operators. Consider the map \(\Phi\) defined as

        \[
            \Phi(X) = X - U X U^*
        \]

        where

        \[
            U = \frac{1}{\sqrt{2}}
            \begin{pmatrix}
                1 & 1 \\
                -1 & 1
            \end{pmatrix}.
        \]

        To check if this is a valid quantum channel or not,

        ```python exec="1" source="above"
        import numpy as np
        from toqito.matrices import pauli
        from toqito.channel_props import is_quantum_channel

        U = (1/np.sqrt(2))*np.array([[1, 1],[-1, 1]])
        X = pauli("X")
        phi = X - np.matmul(U, np.matmul(X, np.conjugate(U)))

        print(is_quantum_channel(phi))
        ```

        If we instead check for the validity of depolarizing channel being a valid quantum channel,

        ```python exec="1" source="above"
        from toqito.channels import depolarizing
        from toqito.channel_props import is_quantum_channel

        choi_depolarizing = depolarizing(dim=2, param_p=0.2)

        print(is_quantum_channel(choi_depolarizing))
        ```

    Args:
        phi: The channel provided as either a Choi matrix or a list of Kraus operators.
        rtol: The relative tolerance parameter (default 1e-05).
        atol: The absolute tolerance parameter (default 1e-08).

    Returns:
        `True` if the channel is a quantum channel, and `False` otherwise.

    """
    # If the variable `phi` is provided as a list, we assume this is a list
    # of Kraus operators.
    if not (
        isinstance(phi, np.ndarray)
        or (
            isinstance(phi, list)
            and all(isinstance(row, list) and all(isinstance(op, np.ndarray) for op in row) for row in phi)
        )
    ):
        raise TypeError(
            "phi must be either a numpy array (Choi matrix) or a list of lists of numpy arrays (Kraus operators)."
        )
    if isinstance(phi, list):
        phi = kraus_to_choi(phi)

    # A valid quantum channel is a superoperator that is both completely
    # positive and trace-preserving.
    try:
        return is_completely_positive(phi, rtol, atol) and is_trace_preserving(phi, rtol, atol)
    except Exception:
        return False
