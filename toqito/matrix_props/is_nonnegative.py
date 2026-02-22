"""Checks if the matrix is nonnegative or doubly nonnegative."""

import numpy as np

from toqito.matrix_props import is_positive_semidefinite


def is_nonnegative(input_mat: np.ndarray, mat_type: str = "nonnegative") -> bool:
    r"""Check if the matrix is nonnegative.

    When all the entries in the matrix are larger than or equal to zero the matrix of interest is a
    nonnegative matrix [@WikiNonNegative].

    When a matrix is nonegative and positive semidefinite [@WikiPosDef], the matrix is doubly nonnegative.


    Examples:
    We expect an identity matrix to be nonnegative.

    ```python exec="1" source="above"
    import numpy as np
    from toqito.matrix_props import is_nonnegative
    
    print(is_nonnegative(np.eye(2)))
    print(is_nonnegative(np.eye(2), "doubly"))
    print(is_nonnegative(np.array([[1, -1], [1, 1]])))
    ```


    Raises:
        TypeError: If something other than `"doubly"`or `"nonnegative"` is used for `mat_type`.

    Args:
        input_mat: np.ndarray Matrix of interest.
        mat_type: Type of nonnegative matrix. `"nonnegative"` for a nonnegative matrix and `"doubly"` for a doubly nonnegative matrix.

    """
    valid_types = {"nonnegative", "doubly"}
    if mat_type not in valid_types:
        raise TypeError(f"Invalid matrix check type: {mat_type}. Must be one of: {valid_types}.")

    is_entrywise_nonnegative = bool(np.all(input_mat >= 0))

    if mat_type == "doubly":
        return is_entrywise_nonnegative and is_positive_semidefinite(input_mat)
    else:
        return is_entrywise_nonnegative
