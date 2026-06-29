"""Applies a real power to a positive semidefinite matrix on its support."""

import numpy as np


def psd_matrix_power(mat: np.ndarray, power: float, tol: float = 1e-12) -> np.ndarray:
    r"""Raise a positive semidefinite matrix to a real power on its support.

    The matrix is diagonalized and each eigenvalue greater than ``tol`` is raised
    to ``power``; eigenvalues at or below ``tol`` are treated as zero and left at
    zero. Restricting to the support keeps negative and fractional powers well
    defined for rank-deficient matrices.

    Args:
        mat: A positive semidefinite matrix.
        power: The real exponent to apply.
        tol: Eigenvalues at or below this threshold are treated as zero.

    Returns:
        The matrix ``mat`` raised to ``power`` on its support.

    Examples:
        The square root of a rank-deficient PSD matrix stays on its support:

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.matrix_ops import psd_matrix_power

        mat = np.array([[4.0, 0.0], [0.0, 0.0]])
        print(psd_matrix_power(mat, 0.5))
        ```

    """
    eigvals, eigvecs = np.linalg.eigh((mat + mat.conj().T) / 2)
    powered = np.zeros_like(eigvals, dtype=float)
    positive = eigvals > tol
    powered[positive] = eigvals[positive] ** power
    return eigvecs @ np.diag(powered) @ eigvecs.conj().T
