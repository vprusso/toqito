"""Generates the depolarizing channel."""

import numpy as np


def depolarizing(dim: int, param_p: float = 0) -> np.ndarray:
    r"""Produce the partially depolarizing channel.

    (Section: Replacement Channels and the Completely Depolarizing Channel from
    [@Watrous_2018_TQI]).

    The Choi matrix of the completely depolarizing channel [@WikiDepo] that acts on
    `dim`-by-`dim` matrices.

    The *completely depolarizing channel* is defined as

    \[
        \Omega(X) = \text{Tr}(X) \omega
    \]

    for all \(X \in \text{L}(\mathcal{X})\), where

    \[
        \omega = \frac{\mathbb{I}_{\mathcal{X}}}{\text{dim}(\mathcal{X})}
    \]

    denotes the completely mixed stated defined with respect to the space \(\mathcal{X}\).

    Examples:
    The completely depolarizing channel maps every density matrix to the maximally-mixed state.
    For example, consider the density operator

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

    ```python exec="1" source="above"
    import numpy as np
    from toqito.channels import depolarizing
    from toqito.channel_ops import apply_channel

    test_input_mat = np.array([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]])

    print(apply_channel(test_input_mat, depolarizing(4)))
    ```

    ```python exec="1" source="above"
    import numpy as np
    from toqito.channels import depolarizing
    from toqito.channel_ops import apply_channel

    test_input_mat = np.arange(1, 17).reshape(4, 4)

    print(apply_channel(test_input_mat, depolarizing(4, 0.5)))
    ```

    Raises:
        ValueError: If `param_p` is outside the interval [0,1].

    Args:
        dim: The dimensionality on which the channel acts.
        param_p: Depolarizing probability \(p \) \in [0,1] that mixes the input state with the maximally mixed state.
        Default 0.

    Returns:
        The Choi matrix of the completely depolarizing channel.

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
