"""Checks if matrix is pseudo hermitian with respect to given signature."""

import numpy as np

from toqito.matrix_props import has_same_dimension, is_hermitian, is_square


def is_pseudo_hermitian(mat: np.ndarray, signature: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    r"""Check if a matrix is pseudo-Hermitian.

    A matrix \(H\) is pseudo-Hermitian with respect to a given signature matrix \(\eta\) if it satisfies:

    \[
        \eta H \eta^{-1} = H^{\dagger},
    \]

    where:

    - \(H^{\dagger}\) is the conjugate transpose (Hermitian transpose) of \(H\),
    - \(\eta\) is a Hermitian, invertible matrix.

    Examples:
        Consider the following matrix:

        \[
            H = \begin{pmatrix}
                1 & 1+i \\
                -1+i & -1
            \end{pmatrix}
        \]

        with the signature matrix:

        \[
            \eta = \begin{pmatrix}
                1 & 0 \\
                0 & -1
            \end{pmatrix}
        \]

        Our function confirms that \(H\) is pseudo-Hermitian:

        ```python exec="1" source="above"
        import numpy as np
        from toqito.matrix_props import is_pseudo_hermitian

        H = np.array([[1, 1+1j], [-1+1j, -1]])
        eta = np.array([[1, 0], [0, -1]])

        print(is_pseudo_hermitian(H, eta))
        ```

        However, the following matrix \(A\)

        \[
            A = \begin{pmatrix}
                1 & i \\
                -i & 1
            \end{pmatrix}
        \]

        is not pseudo-Hermitian with respect to the same signature matrix.

        ```python exec="1" source="above"
        import numpy as np
        from toqito.matrix_props import is_pseudo_hermitian
        eta = np.array([[1, 0], [0, -1]])
        A = np.array([[1, 1j], [-1j, 1]])

        print(is_pseudo_hermitian(A, eta))
        ```

    Raises:
        ValueError: If `signature` is not Hermitian or not invertible.

    Args:
        mat: The matrix to check.
        signature: The signature matrix \(\eta\), which must be Hermitian and invertible.
        rtol: The relative tolerance parameter (default 1e-05).
        atol: The absolute tolerance parameter (default 1e-08).

    Returns:
        Return `True` if the matrix is pseudo-Hermitian, and `False` otherwise.

    """
    if not is_hermitian(signature):
        raise ValueError("Signature not hermitian matrix.")

    if np.linalg.matrix_rank(signature) != signature.shape[0]:
        raise ValueError("Signature is not invertible.")

    if not is_square(mat) or not has_same_dimension([mat, signature]):
        return False

    eta_H_inv_eta = signature @ mat @ np.linalg.inv(signature)
    return np.allclose(eta_H_inv_eta, mat.conj().T, rtol=rtol, atol=atol)
