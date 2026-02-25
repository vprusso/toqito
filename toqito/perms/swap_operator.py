"""Swap operator. is used to generate a unitary operator that can swap two subsystems."""

import numpy as np
from scipy import sparse

from toqito.perms import swap


def swap_operator(dim: list[int] | int, is_sparse: bool = False) -> np.ndarray:
    r"""Produce a unitary operator that swaps two subsystems.

    Provides the unitary operator that swaps two copies of `dim`-dimensional space. If the two subsystems are not
    of the same dimension, `dim` should be a 1-by-2 vector containing the dimension of the subsystems.

    Examples:

    The \(2\)-dimensional swap operator is given by the following matrix

    \[
        X_2 =
        \begin{pmatrix}
            1 & 0 & 0 & 0 \\
            0 & 0 & 1 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & 0 & 1
        \end{pmatrix}
    \]

    Using `|toqito⟩` we can obtain this matrix as follows.

    ```python exec="1" source="above"
    from toqito.perms import swap_operator
    
    print(swap_operator(2))
    ```

    The \(3\)-dimensional operator may be obtained using `|toqito⟩` as follows.

    ```python exec="1" source="above"
    from toqito.perms import swap_operator
    
    print(swap_operator(3))
    ```



    Args:
        dim: The dimensions of the subsystems.
        is_sparse: Sparse if `True` and non-sparse if `False`.

    Returns:
        The swap operator of dimension `dim`.

    """
    # Allow the user to enter a single number for dimension.
    if isinstance(dim, int):
        dim = np.array([dim, dim])

    mat = sparse.identity(int(np.prod(dim))) if is_sparse else np.identity(int(np.prod(dim)))
    # Swap the rows of the identity appropriately.
    return swap(rho=mat, sys=[1, 2], dim=dim, row_only=True)
