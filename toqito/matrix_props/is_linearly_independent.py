"""Checks if a set of vectors are linearly independent."""

import numpy as np


def is_linearly_independent(vectors: list[np.ndarray]) -> bool:
    r"""Check if set of vectors are linearly independent [@WikiLinearIndependence].

    Examples:
    The following vectors are an example of a linearly independent set of vectors in \(\mathbb{R}^3\).

    \[
        \begin{pmatrix}
            1 \\ 0 \\ 1
        \end{pmatrix}, \quad
        \begin{pmatrix}
            1 \\ 1 \\ 0
        \end{pmatrix}, \quad \text{and} \quad
        \begin{pmatrix}
            0 \\ 0 \\ 1
        \end{pmatrix}
    \]

    We can see that these are linearly independent.

    ```python exec="1" source="above"
    import numpy as np
    from toqito.matrix_props import is_linearly_independent

    v_1 = np.array([[1], [0], [1]])
    v_2 = np.array([[1], [1], [0]])
    v_3 = np.array([[0], [0], [1]])

    print(is_linearly_independent([v_1, v_2, v_3]))
    ```

    Args:
        vectors: Vectors to check the linear independence of.

    Returns:
        Return `True` if vectors are linearly independent `False` otherwise.

    """
    # Check if the rank of the matrix equals the number of vectors.
    return bool(np.linalg.matrix_rank(np.column_stack(vectors)) == len(vectors))
