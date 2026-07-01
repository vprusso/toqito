"""Checks if the matrix is an isometry."""

import numpy as np


def is_isometry(mat: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    r"""Check if a matrix is an isometry [@wikipediaisometry].

    A matrix \(V \in \text{L}(\mathcal{X}, \mathcal{Y})\) is an isometry if it preserves inner
    products, equivalently if its columns are orthonormal, that is if

    \[
        V^* V = \mathbb{I}_{\mathcal{X}},
    \]

    where \(\mathbb{I}_{\mathcal{X}}\) is the identity on the input space. The matrix need not be
    square; this requires \(\dim(\mathcal{Y}) \geq \dim(\mathcal{X})\). A square isometry is a unitary
    matrix.

    Args:
        mat: Matrix to check.
        rtol: The relative tolerance parameter (default 1e-05).
        atol: The absolute tolerance parameter (default 1e-08).

    Returns:
        Return `True` if the matrix is an isometry, and `False` otherwise.

    Examples:
        The columns of the following \(3 \times 2\) matrix are orthonormal, so it is an isometry that
        embeds \(\mathbb{C}^2\) into \(\mathbb{C}^3\):

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.matrix_props import is_isometry

        V = np.array([[1, 0], [0, 1], [0, 0]])

        print(is_isometry(V))
        ```

        Every unitary matrix is a (square) isometry:

        ```python exec="1" source="above" result="text"
        from toqito.matrix_props import is_isometry
        from toqito.rand import random_unitary

        print(is_isometry(random_unitary(3)))
        ```

        A matrix whose columns are not orthonormal is not an isometry:

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.matrix_props import is_isometry

        B = np.array([[1, 0], [1, 1]])

        print(is_isometry(B))
        ```

    """
    dim_in = mat.shape[1]
    vc_v_mat = mat.conj().T @ mat

    # If V^* V = I on the input space, the columns of V are orthonormal, so V is an isometry.
    return bool(np.allclose(vc_v_mat, np.eye(dim_in), rtol=rtol, atol=atol))
