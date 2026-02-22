"""Checks if the matrix is positive."""

import numpy as np


def is_positive(input_mat: np.ndarray) -> bool:
    r"""Check if the matrix is positive.

    When all the entries in the matrix are larger than zero the matrix of interest is a
    positive matrix [@WikiNonNegative].

    !!! note
        This function is different from [`is_positive_definite`][toqito.matrix_props.is_positive_definite],
        [`is_totally_positive`][toqito.matrix_props.is_totally_positive] and [`is_positive_semidefinite`][toqito.matrix_props.is_positive_semidefinite].

    Examples:
    We expect a matrix full of 1s to be positive.

    ```python exec="1" source="above"
    import numpy as np
    from toqito.matrix_props import is_positive
    
    input_mat = np.array([[1, 1, 1],[1, 1, 1],[1, 1, 1]])
    
    print(is_positive(input_mat))
    ```

    Args:
        input_mat: Matrix of interest.

    """
    return bool(np.all(input_mat > 0))
