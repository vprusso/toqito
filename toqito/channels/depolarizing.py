"""Generates the depolarizing channel."""

import numpy as np


def depolarizing(dim: int, param_p: float = 0) -> np.ndarray:
    r"""Produce the partially depolarizing channel.

    (Section: Replacement Channels and the Completely Depolarizing Channel from
    [@watrous2018theory]).

    The Choi matrix of the partially depolarizing channel [@wikipediadepolarizing] that acts on
    `dim`-by-`dim` matrices.

    The *partially depolarizing channel* is defined as

    \[
        \Phi_p(\rho) = (1 - p) \text{Tr}(\rho) \frac{\mathbb{I}}{d} + p \, \rho
    \]

    for all \(\rho \in \text{L}(\mathcal{X})\), where \(d = \text{dim}(\mathcal{X})\)
    and \(p \in [0, 1]\).

    When \(p = 0\), this reduces to the *completely depolarizing channel*
    \(\Omega(\rho) = \text{Tr}(\rho) \frac{\mathbb{I}}{d}\), which maps every state to the
    maximally mixed state. When \(p = 1\), this is the identity channel.

    The corresponding Choi matrix is

    \[
        J(\Phi_p) = \frac{1 - p}{d} \, \mathbb{I} \otimes \mathbb{I}
        + p \, |\psi\rangle\!\langle\psi|
    \]

    where \(|\psi\rangle = \sum_{i} |i\rangle \otimes |i\rangle\) is the (unnormalized)
    maximally entangled state.

    Note:
        This follows the QETLAB convention where \(p = 0\) gives the completely depolarizing
        channel and \(p = 1\) gives the identity channel.

    Args:
        dim: The dimensionality on which the channel acts.
        param_p: Parameter \(p \in [0, 1]\) that interpolates between the completely depolarizing
            channel (\(p = 0\)) and the identity channel (\(p = 1\)). Default 0.

    Returns:
        The Choi matrix of the partially depolarizing channel.

    Raises:
        ValueError: If `param_p` is outside the interval [0,1].

    Examples:
        The completely depolarizing channel (\(p = 0\)) maps every density matrix to the
        maximally-mixed state. For example, consider the density operator

        \[
            \rho = \frac{1}{2} \begin{pmatrix}
                                 1 & 0 & 0 & 1 \\
                                 0 & 0 & 0 & 0 \\
                                 0 & 0 & 0 & 0 \\
                                 1 & 0 & 0 & 1
                               \end{pmatrix}
        \]

        corresponding to one of the Bell states. Applying the depolarizing channel to \(\rho\) we
        have that

        \[
            \Phi(\rho) = \frac{1}{4} \begin{pmatrix}
                                        1 & 0 & 0 & 0 \\
                                        0 & 1 & 0 & 0 \\
                                        0 & 0 & 1 & 0 \\
                                        0 & 0 & 0 & 1
                                     \end{pmatrix}.
        \]

        This can be observed in `|toqito⟩` as follows.

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.channels import depolarizing
        from toqito.channel_ops import apply_channel

        test_input_mat = np.array([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]])

        print(apply_channel(test_input_mat, depolarizing(4)))
        ```

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.channels import depolarizing
        from toqito.channel_ops import apply_channel

        test_input_mat = np.arange(1, 17).reshape(4, 4)

        print(apply_channel(test_input_mat, depolarizing(4, 0.5)))
        ```

    """
    # Compute the Choi matrix of the depolarizing channel.
    if param_p > 1 or param_p < 0:
        raise ValueError("The depolarizing probability must be between 0 and 1.")

    result = np.zeros((dim**2, dim**2), dtype=np.float64)
    np.fill_diagonal(result, (1 - param_p) / dim)

    if param_p != 0.0:
        idx = np.arange(dim) * (dim + 1)
        result[np.ix_(idx, idx)] += param_p

    return result
