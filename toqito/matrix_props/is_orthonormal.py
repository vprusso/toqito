"""Checks if the set of vectors are orthonormal."""

import numpy as np


def is_orthonormal(vectors: list[np.ndarray]) -> bool:
    r"""Check if the vectors are orthonormal.

    Args:
        vectors: A list of `np.ndarray` vectors, each given as a 1-D array or a column/row vector.

    Returns:
        True if vectors are orthonormal; False otherwise.

    Examples:
        The following vectors are an example of an orthonormal set of
        vectors in \(\mathbb{R}^3\).

        \[
            \begin{pmatrix}
                1 \\ 0 \\ 0
            \end{pmatrix}, \quad
            \begin{pmatrix}
                0 \\ 1 \\ 0
            \end{pmatrix}, \quad \text{and} \quad
            \begin{pmatrix}
                0 \\ 0 \\ 1
            \end{pmatrix}
        \]

        To check these are a known set of orthonormal vectors:

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.matrix_props import is_orthonormal

        v_1 = np.array([1, 0, 0])
        v_2 = np.array([0, 1, 0])
        v_3 = np.array([0, 0, 1])

        v = [v_1, v_2, v_3]

        print(is_orthonormal(v))
        ```

    """
    # Flatten each vector to 1-D so column vectors (n, 1) and row vectors (1, n) are handled the
    # same as plain 1-D arrays; the Gram matrix of an orthonormal set is the identity.
    mat = np.array([np.asarray(vec).reshape(-1) for vec in vectors])
    return np.allclose(mat @ mat.conj().T, np.eye(len(vectors)))
