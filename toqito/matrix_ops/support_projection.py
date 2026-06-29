"""Constructs the orthogonal projector onto the support of a PSD matrix."""

import numpy as np


def support_projection(mat: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    r"""Return the orthogonal projector onto the support of a PSD matrix.

    The support is the span of the eigenvectors whose eigenvalues exceed ``tol``.
    If no eigenvalue exceeds ``tol`` (for instance, the zero matrix), the zero
    projector is returned.

    Args:
        mat: A positive semidefinite matrix.
        tol: Eigenvalues at or below this threshold are treated as zero.

    Returns:
        The orthogonal projector onto the support of ``mat``.

    Examples:
        The support projector of a rank-one PSD matrix is the projector onto its
        range:

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.matrix_ops import support_projection

        mat = np.array([[2.0, 0.0], [0.0, 0.0]])
        print(support_projection(mat))
        ```

    """
    eigvals, eigvecs = np.linalg.eigh((mat + mat.conj().T) / 2)
    positive = eigvals > tol
    if not np.any(positive):
        return np.zeros_like(mat, dtype=complex)
    return eigvecs[:, positive] @ eigvecs[:, positive].conj().T
