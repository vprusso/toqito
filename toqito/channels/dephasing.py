"""Generates the dephasing channel."""

import numpy as np

from toqito.states import max_entangled


def dephasing(dim: int, param_p: float = 0) -> np.ndarray:
    r"""Produce the partially dephasing channel.

    (Section: The Completely Dephasing Channel from [@Watrous_2018_TQI]).

    The Choi matrix of the completely dephasing channel that acts on `dim`-by-`dim`
    matrices.

    Let \(\Sigma\) be an alphabet and let \(\mathcal{X} = \mathbb{C}^{\Sigma}\). The map
    \(\Delta \in \text{T}(\mathcal{X})\) defined as

    \[
        \Delta(X) = \sum_{a \in \Sigma} X(a, a) E_{a,a}
    \]

    for every \(X \in \text{L}(\mathcal{X})\) is defined as the *completely dephasing channel*.

    Examples:
    The completely dephasing channel maps kills everything off the diagonal. Consider the
    following matrix

    \[
        \rho = \begin{pmatrix}
                   1 & 2 & 3 & 4 \\
                   5 & 6 & 7 & 8 \\
                   9 & 10 & 11 & 12 \\
                   13 & 14 & 15 & 16
               \end{pmatrix}.
    \]

    Applying the dephasing channel to \(\rho\) we have that

    \[
        \Phi(\rho) = \begin{pmatrix}
                         1 & 0 & 0 & 0 \\
                         0 & 6 & 0 & 0 \\
                         0 & 0 & 11 & 0 \\
                         0 & 0 & 0 & 16
                     \end{pmatrix}.
    \]

    This can be observed in `|toqito⟩` as follows.

    ```python exec="1" source="above"
    import numpy as np
    from toqito.channels import dephasing
    from toqito.channel_ops import apply_channel

    test_input_mat = np.arange(1, 17).reshape(4, 4)

    print(apply_channel(test_input_mat, dephasing(4)))
    ```


    We may also consider setting the parameter `p = 0.5`.

    ```python exec="1" source="above"
    import numpy as np
    from toqito.channels import dephasing
    from toqito.channel_ops import apply_channel

    test_input_mat = np.arange(1, 17).reshape(4, 4)

    print(apply_channel(test_input_mat, dephasing(4, 0.5)))
    ```

    Args:
        dim: The dimensionality on which the channel acts.
        param_p: Default is 0.

    Returns:
        The Choi matrix of the dephasing channel.

    """
    # Compute the Choi matrix of the dephasing channel.

    # Gives a sparse non-normalized state.
    psi = max_entangled(dim=dim, is_sparse=False, is_normalized=False)
    return (1 - param_p) * np.diag(np.diag(psi @ psi.conj().T)) + param_p * (psi @ psi.conj().T)
