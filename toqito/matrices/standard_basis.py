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

    This function was inspired by [@Seshadri_2021_Git, Seshadri_2021_Theory, Seshadri_2021_Versatile]

    Examples:
        ```python exec="1" source="above"
        from toqito.matrices import standard_basis

        print(standard_basis(2))
        ```

    Args:
        dim: The dimension of the basis.
        flatten: If True, the basis is returned as a flattened list.

    Returns:
        A list of numpy.ndarray of shape (n, 1).

    """
    first_basis_vector = np.zeros(dim) if flatten else np.zeros((dim, 1))
    first_basis_vector[0] = 1.0

    # The standard_basis is obtained by cyclic permutations of the first basis
    # vector
    return [np.array([first_basis_vector[i - j] for i in range(dim)]) for j in range(dim)]
