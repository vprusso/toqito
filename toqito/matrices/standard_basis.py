"""Constructs the standard basis."""

import numpy as np


def standard_basis(dim: int, flatten: bool = False) -> list[np.ndarray]:
    r"""Create standard basis of dimension `dim`.

    Create a list containing the elements of the standard basis for the
    given dimension:

    \[
        |1> = (1, 0, 0, ..., 0)^T
        |2> = (0, 1, 0, ..., 0)^T
        .
        .
        .
        |n> = (0, 0, 0, ..., 1)^T
    \]

    This function was inspired by [@seshadri2021git;@seshadri2021theory;@seshadri2021versatile]

    Args:
        dim: The dimension of the basis.
        flatten: If True, the basis is returned as a flattened list.

    Returns:
        A list of numpy.ndarray of shape (n, 1).

    Examples:
        ```python exec="1" source="above" result="text"
        from toqito.matrices import standard_basis

        print(standard_basis(2))
        ```

    """
    basis = np.eye(dim)
    if not flatten:
        basis = basis[..., np.newaxis]
    return list(basis)
