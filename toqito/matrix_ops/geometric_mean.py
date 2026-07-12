"""Calculates the t-weighted matrix geometric mean of two matrices."""

import numpy as np

from toqito.cones._utils import _require_square_2d


def geometric_mean(mat_a: np.ndarray, mat_b: np.ndarray, t: float) -> np.ndarray:
    r"""Calculate the t-weighted matrix geometric mean of two matrices [@fawzi2015matrixgeometric].

    Since the inputs are positive definite, the matrix geometric mean `G_t(A, B)`
    is defined by the following equation:

    \[
        G_t(A, B) = A^{1/2} (A^{-1/2} B A^{-1/2})^{t} A^{1/2}.
    \]

    If `A` and `B` commute, then this reduces to

    \[
        G_t(A, B) = A^{1-t} B^{t}.
    \]

    The return value of this function is a matrix of the same size as the inputs.

    Args:
      mat_a: A positive definite matrix.
      mat_b: A positive definite matrix.
      t: The weight in the range `[-1, 2]` (includes `[0, 1]` used by the hypo cone).

    Raises:
      ValueError: If the matrices are not the same size, not 2D, or not square.
      ValueError: If the matrices are not positive definite.
      ValueError: If the weight is not in the range `[-1, 2]`.

    Returns:
      A matrix with the same shape as the inputs.

    Examples:
        For the commuting positive definite matrices \(A = \text{diag}(2, 4)\) and
        \(B = \text{diag}(8, 16)\), the weighted mean reduces to \(G_t(A, B) = A^{1-t}B^t\),
        so for \(t = 1/2\) the result is \(\text{diag}(\sqrt{2 \cdot 8}, \sqrt{4 \cdot 16})\):

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.matrix_ops import geometric_mean

        mat_a = np.diag([2.0, 4.0])
        mat_b = np.diag([8.0, 16.0])
        print(geometric_mean(mat_a, mat_b, 1 / 2))
        ```

    """
    if mat_a.shape != mat_b.shape:
        raise ValueError("The matrices must be the same size.")
    _require_square_2d(mat_a, "The matrices")
    if t < -1 or t > 2:
        raise ValueError(f"The weight t must be in the range [-1, 2]; got {t}.")
    # Deferred import: matrix_ops must stay free of a matrix_props back-edge (avoids a
    # load-time import cycle with state_props/channels).
    from toqito.matrix_props.is_positive_definite import is_positive_definite  # noqa: PLC0415

    if not is_positive_definite(mat_a) or not is_positive_definite(mat_b):
        raise ValueError("The matrices must be positive definite.")

    eigvals_a, eigvecs_a = np.linalg.eigh(mat_a)
    sqrt_eigvals = np.sqrt(eigvals_a)
    sqrt_a = (eigvecs_a * sqrt_eigvals) @ eigvecs_a.conj().T
    a_inv_sqrt = (eigvecs_a * (1 / sqrt_eigvals)) @ eigvecs_a.conj().T

    middle_term = a_inv_sqrt @ mat_b @ a_inv_sqrt
    middle_term = (middle_term + middle_term.conj().T) / 2
    eigvals_mid, eigvecs_mid = np.linalg.eigh(middle_term)
    middle_pow = (eigvecs_mid * eigvals_mid**t) @ eigvecs_mid.conj().T
    return sqrt_a @ middle_pow @ sqrt_a
