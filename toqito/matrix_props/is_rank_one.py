"""Checks if a matrix has rank one."""

import numpy as np


def is_rank_one(mat: np.ndarray, tol: float = 1e-08) -> bool:
    r"""Determine whether the given matrix has rank one [@WikiMatrixRank].

    The function evaluates the singular values (equivalently, eigenvalues for Hermitian matrices)
    and counts how many are greater than the provided tolerance.

    Examples:

    Consider the Bell state density matrix \(\rho = \ket{\Phi^+}\bra{\Phi^+}\). This matrix
    has rank one.

    ```python exec="1" source="above"
    from toqito.matrix_props import is_rank_one
    from toqito.states import bell
    
    rho = bell(0) @ bell(0).conj().T
    print(is_rank_one(rho))
    ```

    On the other hand, the maximally mixed state is not rank one.

    ```python exec="1" source="above"
    import numpy as np
    from toqito.matrix_props import is_rank_one
    
    maximally_mixed = np.eye(2) / 2
    print(is_rank_one(maximally_mixed))
    ```

    Args:
        mat: Matrix to test.
        tol: Numerical tolerance used when distinguishing non-zero singular values.

    Returns:
        `True` if the matrix has rank at most one, `False` otherwise.

    """
    singular_values = np.linalg.svd(mat, compute_uv=False)
    return np.count_nonzero(singular_values > tol) <= 1
