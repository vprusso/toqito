"""Generates the dephasing channel."""

import numpy as np

from toqito.states import max_entangled


def dephasing(dim: int, param_p: float = 0) -> np.ndarray:
    r"""Produce the partially dephasing channel.

    (Section: The Completely Dephasing Channel from [@watrous2018theory]).

    The Choi matrix of the partially dephasing channel that acts on `dim`-by-`dim`
    matrices.

    The *partially dephasing channel* is defined as

    \[
        \Phi_p(\rho) = (1 - p) \, \Delta(\rho) + p \, \rho
    \]

    where \(\Delta\) is the *completely dephasing channel*

    \[
        \Delta(X) = \sum_{a \in \Sigma} X(a, a) E_{a,a}
    \]

    which removes all off-diagonal elements. Here \(p \in [0, 1]\).

    When \(p = 0\), this reduces to the completely dephasing channel \(\Delta\).
    When \(p = 1\), this is the identity channel.

    The corresponding Choi matrix is

    \[
        J(\Phi_p) = (1 - p) \, \text{diag}(|\psi\rangle\!\langle\psi|)
        + p \, |\psi\rangle\!\langle\psi|
    \]

    where \(|\psi\rangle = \sum_{i} |i\rangle \otimes |i\rangle\) is the (unnormalized)
    maximally entangled state.

    Args:
        dim: The dimensionality on which the channel acts.
        param_p: Parameter \(p \in [0, 1]\) that interpolates between the completely dephasing
            channel (\(p = 0\)) and the identity channel (\(p = 1\)). Default 0.

    Returns:
        The Choi matrix of the partially dephasing channel.

    Examples:
        The completely dephasing channel (\(p = 0\)) kills everything off the diagonal. Consider
        the following matrix

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

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.channels import dephasing
        from toqito.channel_ops import apply_channel

        test_input_mat = np.arange(1, 17).reshape(4, 4)

        print(apply_channel(test_input_mat, dephasing(4)))
        ```


        We may also consider setting the parameter `p = 0.5`.

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.channels import dephasing
        from toqito.channel_ops import apply_channel

        test_input_mat = np.arange(1, 17).reshape(4, 4)

        print(apply_channel(test_input_mat, dephasing(4, 0.5)))
        ```

    """
    # Compute the Choi matrix of the dephasing channel.

    # Gives a sparse non-normalized state.
    psi = max_entangled(dim=dim, is_sparse=False, is_normalized=False)
    return (1 - param_p) * np.diag(np.diag(psi @ psi.conj().T)) + param_p * (psi @ psi.conj().T)
