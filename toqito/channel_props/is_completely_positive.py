"""Determines if a channel is completely positive."""

import numpy as np

from toqito.channel_ops import kraus_to_choi
from toqito.channel_props import is_herm_preserving
from toqito.matrix_props import is_positive_semidefinite


def is_completely_positive(
    phi: np.ndarray | list[list[np.ndarray]],
    rtol: float = 1e-05,
    atol: float = 1e-08,
) -> bool:
    r"""Determine whether the given channel is completely positive.

    (Section: Linear Maps Of Square Operators from [@Watrous_2018_TQI]).

    A map \(\Phi \in \text{T} \left(\mathcal{X}, \mathcal{Y} \right)\) is *completely
    positive* if it holds that

    \[
        \Phi \otimes \mathbb{I}_{\text{L}(\mathcal{Z})}
    \]

    is a positive map for every complex Euclidean space \(\mathcal{Z}\).

    Alternatively, a channel is completely positive if the corresponding Choi matrix of the
    channel is both Hermitian-preserving and positive semidefinite.

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

        This map is not completely positive, as we can verify as follows.

        ```python exec="1" source="above"
        import numpy as np
        from toqito.channel_props import is_completely_positive

        unitary_mat = np.array([[1, 1], [-1, 1]]) / np.sqrt(2)
        kraus_ops = [[np.identity(2), np.identity(2)], [unitary_mat, -unitary_mat]]

        print(is_completely_positive(kraus_ops))
        ```

        We can also specify the input as a Choi matrix. For instance, consider the Choi matrix
        corresponding to the \(2\)-dimensional completely depolarizing channel

        \[
            \Omega =
            \frac{1}{2}
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 0 & 0 & 1
            \end{pmatrix}.
        \]

        We may verify that this channel is completely positive

        ```python exec="1" source="above"
        from toqito.channels import depolarizing
        from toqito.channel_props import is_completely_positive

        print(is_completely_positive(depolarizing(2)))
        ```

    Args:
        phi: The channel provided as either a Choi matrix or a list of Kraus operators.
        rtol: The relative tolerance parameter (default 1e-05).
        atol: The absolute tolerance parameter (default 1e-08).

    Returns:
        True if the channel is completely positive, and False otherwise.

    """
    # If the variable `phi` is provided as a list, we assume this is a list
    # of Kraus operators.
    if isinstance(phi, list):
        phi = kraus_to_choi(phi)

    # Use Choi's theorem to determine whether `phi` is completely positive.
    return is_herm_preserving(phi, rtol, atol) and is_positive_semidefinite(phi, rtol, atol)
