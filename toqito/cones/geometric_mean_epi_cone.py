"""CVXPY constraints for the matrix geometric mean epigraph cone."""

# Adapted from CVXQUAD (https://github.com/hfawzi/cvxquad), BSD-2-Clause.
# Original implementation by Fawzi, Saunderson, et al.

import cvxpy

from toqito.cones._utils import _require_square_2d, _symmetric_like_variable
from toqito.cones.geometric_mean_hypo_cone import geometric_mean_hypo_cone


def geometric_mean_epi_cone(
    A: cvxpy.Expression,
    B: cvxpy.Expression,
    T: cvxpy.Expression,
    t: float,
    *,
    hermitian: bool = False,
) -> list[cvxpy.Constraint]:
    r"""Return CVX constraints for the matrix geometric-mean epigraph cone [@fawzi2015matrixgeometric].

    The set of matrices that satisfy the constraints are `A`, `B`, `T` triples such
    that

    \[
    G_t(A, B) \leq T
    \]

    where `G_t(A, B)` is the matrix geometric mean function.

    Args:
        A: A cvxpy expression representing a matrix.
        B: A cvxpy expression representing a matrix.
        T: A cvxpy expression representing a matrix.
        t: The weight in the range `[-1, 0]` or `[1, 2]`.
        hermitian: Whether the matrices are Hermitian or symmetric.

    Raises:
        ValueError: If the weight is not in the range `[-1, 0]` or `[1, 2]`.
        ValueError: If the matrices are not the same size.
        ValueError: If the matrices are not 2D or not square.

    Returns:
        A list of CVX constraints.

    """
    if t < -1 or (t > 0 and t < 1) or t > 2:
        raise ValueError("The weight must be in the range [-1, 0] or [1, 2].")

    if A.shape != B.shape or B.shape != T.shape:
        raise ValueError("The matrices must be the same size.")
    _require_square_2d(A, "The matrices")

    dim = A.shape[0]
    z_var = _symmetric_like_variable(dim, hermitian=hermitian)

    if t <= 0:
        lmi = cvxpy.bmat([[T, A], [A, z_var]]) >> 0
        hypo_z = geometric_mean_hypo_cone(A, B, z_var, float(-t), fullhyp=False, hermitian=hermitian)
        return [lmi, *hypo_z]
    lmi = cvxpy.bmat([[T, B], [B, z_var]]) >> 0
    hypo_z = geometric_mean_hypo_cone(A, B, z_var, float(2 - t), fullhyp=False, hermitian=hermitian)
    return [lmi, *hypo_z]
