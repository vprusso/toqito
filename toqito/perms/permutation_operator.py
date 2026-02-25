"""Permutation operator is a unitary operator that permutes subsystems."""

import numpy as np
import scipy as sp

from toqito.perms import permute_systems


def permutation_operator(
    dim: list[int] | int,
    perm: list[int],
    inv_perm: bool = False,
    is_sparse: bool = False,
) -> np.ndarray:
    r"""Produce a unitary operator that permutes subsystems.

    Generates a unitary operator that permutes the order of subsystems according to the permutation vector `perm`,
    where the \(i^{th}\) subsystem has dimension `dim[i]`.

    If `inv_perm` = True, it implements the inverse permutation of `perm`. The permutation operator return
    is full is `is_sparse` is `False` and sparse if `is_sparse` is `True`.

    Examples:

    The permutation operator obtained with dimension \(d = 2\) is equivalent to the standard swap operator on two
    qubits

    \[
        P_{2, [1, 0]} =
        \begin{pmatrix}
            1 & 0 & 0 & 0 \\
            0 & 0 & 1 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & 0 & 1
        \end{pmatrix}
    \]

    Using `|toqito⟩`, this can be achieved in the following manner.

    ```python exec="1" source="above"
    from toqito.perms import permutation_operator
    
    print(permutation_operator(2, [1, 0]))
    ```


    Args:
        dim: The dimensions of the subsystems to be permuted.
        perm: A permutation vector.
        inv_perm: Boolean dictating if `perm` is inverse or not.
        is_sparse: Boolean indicating if return is sparse or not.

    Returns:
        Permutation operator of dimension `dim`.

    """
    # Allow the user to enter a single number for `dim`.
    if isinstance(dim, int):
        dim_arr = np.array([dim] * np.ones(max(perm) + 1))
    elif isinstance(dim, list):
        dim_arr = np.array(dim)
    else:
        dim_arr = dim

    mat = sp.sparse.identity(int(np.prod(dim_arr))) if is_sparse else np.identity(int(np.prod(dim_arr)))
    # Swap the rows of the identity matrix appropriately.

    return permute_systems(mat, perm, dim_arr, True, inv_perm)
