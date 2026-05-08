"""Calculates the t-weighted matrix geometric mean of two matrices."""

import numpy as np
from scipy.linalg import fractional_matrix_power, sqrtm

from toqito.matrix_props.is_positive_definite import is_positive_definite


def matrix_geo_mean(A: np.ndarray, B: np.ndarray, t: float) -> np.ndarray:
    r"""Calculate the t-weighted matrix geometric mean of two matrices [@fawzi2015matrixgeometric].

    Since the inputs are positive definite, the matrix geometric mean `G_t(A, B)`
    is defined by the following equation:

    \[
        G_t(A, B) = A^{1/2} (A^{-1/2} B A^{-1/2})^{t} A^{1/2}.
    \]

    If the input matrices are scalars or they commute, then this reduces to

    \[
        G_t(A, B) = A^{1-t} B^{t}.
    \]

    The return value of this function is a matrix of the same size as the inputs.

    Args:
      A: A positive definite matrix.
      B: A positive definite matrix.
      t: The weight in the range `[-1, 2]` (includes `[0, 1]` used by the hypo cone).

    Raises:
      ValueError: If the matrices are not the same size, not 2D, or not square.
      ValueError: If the matrices are not positive definite.
      ValueError: If the weight is not in the range [-1, 2].

    Returns:
      The `t`-weighted matrix geometric mean of `A` and `B`.

    """
    if A.shape != B.shape:
        raise ValueError("The matrices must be the same size.")
    if A.ndim != 2:
        raise ValueError("The matrices must be 2D arrays.")
    if A.shape[0] != A.shape[1]:
        raise ValueError("The matrices must be square.")
    if t < -1 or t > 2:
        raise ValueError("The weight must be in the range [-1, 2].")
    if not is_positive_definite(A) or not is_positive_definite(B):
        raise ValueError("The matrices must be positive definite.")

    sqrt_A = sqrtm(A)
    A_inv_sqrt = fractional_matrix_power(A, -1 / 2)
    middle_term = A_inv_sqrt @ B @ A_inv_sqrt
    return sqrt_A @ fractional_matrix_power(middle_term, t) @ sqrt_A
