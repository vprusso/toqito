"""Calculates the t-weighted matrix geometric mean of two matrices."""

import numpy as np
from scipy.linalg import fractional_matrix_power, sqrtm

from toqito.matrix_ops._cone_utils import _require_square_2d
from toqito.matrix_props.is_positive_definite import is_positive_definite


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
      ValueError: If the weight is not in the range [-1, 2].

    Returns:
      A matrix with the same shape as the inputs.

    """
    if mat_a.shape != mat_b.shape:
        raise ValueError("The matrices must be the same size.")
    _require_square_2d(mat_a, "The matrices")
    if t < -1 or t > 2:
        raise ValueError("The weight must be in the range [-1, 2].")
    if not is_positive_definite(mat_a) or not is_positive_definite(mat_b):
        raise ValueError("The matrices must be positive definite.")

    sqrt_a = sqrtm(mat_a)
    a_inv_sqrt = fractional_matrix_power(mat_a, -1 / 2)
    middle_term = a_inv_sqrt @ mat_b @ a_inv_sqrt
    return sqrt_a @ fractional_matrix_power(middle_term, t) @ sqrt_a
