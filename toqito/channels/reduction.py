"""Generates the reduction channel."""

import numpy as np
from scipy.sparse import identity

from toqito.states import max_entangled


def reduction(dim: int, k: int = 1) -> np.ndarray:
    r"""Produce the reduction map or reduction channel [@wikipediareduction].

    If `k = 1`, this returns the Choi matrix of the reduction map which is a positive map
    on `dim`-by-`dim` matrices. For a different value of `k`, this yields the
    Choi matrix of the map defined by:

    \[
        R(X) = k * \text{Tr}(X) * \mathbb{I} - X,
    \]

    where \(\mathbb{I}\) is the identity matrix. This map is \(k\)-positive.

    Examples:
        Using `|toqito⟩`, we can generate the \(3\)-dimensional (or standard) reduction map
        as follows.


        ```python exec="1" source="above"
        from toqito.channels import reduction

        print(reduction(3))
        ```

    Args:
        dim: A positive integer (the dimension of the reduction map).
        k: If this positive integer is provided, the script will instead return the Choi matrix of the following
            linear map: Phi(X) := K * Tr(X)I - X.

    Returns:
        The reduction map.

    """
    psi = max_entangled(dim, False, False)
    identity_matrix = identity(dim**2)
    return k * identity_matrix.toarray() - psi @ psi.conj().T
