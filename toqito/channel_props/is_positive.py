"""Determines if a channel is positive."""

import numpy as np

from toqito.channel_ops import kraus_to_choi
from toqito.matrix_props import is_block_positive, is_hermitian


def is_positive(
    phi: np.ndarray | list[list[np.ndarray]],
    rtol: float = 1e-05,
    atol: float = 1e-08,
) -> bool:
    r"""Determine whether the given channel is positive.

    (Section: Linear Maps Of Square Operators from [@watrous2018theory]).

    A map \(\Phi \in \text{T} \left(\mathcal{X}, \mathcal{Y} \right)\) is *positive* if it
    holds that

    \[
        \Phi(P) \in \text{Pos}(\mathcal{Y})
    \]

    for every positive semidefinite operator \(P \in \text{Pos}(\mathcal{X})\).

    A map is positive if and only if its Choi matrix is block positive (i.e. 1-block positive).
    This is strictly weaker than complete positivity (which corresponds to the Choi matrix being
    positive semidefinite), so a positive-but-not-completely-positive map such as the transpose
    map is correctly reported as positive here while `is_completely_positive` reports it as not
    completely positive.

    Note:
        Deciding block positivity is co-NP-hard in general; this routes through
        [`is_block_positive`][toqito.matrix_props.is_block_positive.is_block_positive], which uses
        an S(k)-norm computation and may raise `RuntimeError` for borderline operators it cannot
        resolve.

    Args:
        phi: The channel provided as either a Choi matrix or a list of Kraus operators.
        rtol: The relative tolerance parameter (default 1e-05).
        atol: The absolute tolerance parameter (default 1e-08).

    Returns:
        True if the channel is positive, and False otherwise.

    Examples:
        The transpose map is positive but not completely positive. Its Choi matrix is the swap
        operator, so we can verify that it is positive even though it is not completely positive.

        ```python exec="1" source="above" result="text"
        from toqito.channel_props import is_positive, is_completely_positive
        from toqito.perms import swap_operator

        transpose_choi = swap_operator(2)
        print(is_positive(transpose_choi), is_completely_positive(transpose_choi))
        ```

        We can also specify the input as a Choi matrix. For instance, consider the Choi matrix
        corresponding to the \(4\)-dimensional completely depolarizing channel and may verify
        that this channel is positive.

        ```python exec="1" source="above" result="text"
        from toqito.channels import depolarizing
        from toqito.channel_props import is_positive

        print(is_positive(depolarizing(4)))
        ```

    """
    # If the variable `phi` is provided as a list, we assume this is a list
    # of Kraus operators.
    if isinstance(phi, list):
        phi = kraus_to_choi(phi)
    # A positive map is Hermiticity preserving, so its Choi matrix must be Hermitian.
    if not is_hermitian(phi, rtol=rtol, atol=atol):
        return False
    # The map is positive iff its Choi matrix is block positive (1-block positive).
    return is_block_positive(phi, k=1, rtol=rtol)
