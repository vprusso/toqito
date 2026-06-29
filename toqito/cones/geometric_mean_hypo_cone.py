"""CVXPY constraints for the matrix geometric mean hypograph cone."""

# Adapted from CVXQUAD (https://github.com/hfawzi/cvxquad), BSD-2-Clause.
# Original implementation by Fawzi, Saunderson, et al.

import cvxpy
import numpy as np

from toqito.cones._utils import (
    _is_power_of_two,
    _reduced_fraction_pq,
    _require_square_2d,
    _symmetric_like_variable,
)


def geometric_mean_hypo_cone(
    A: cvxpy.Expression,
    B: cvxpy.Expression,
    T: cvxpy.Expression,
    t: float = 1 / 2,
    fullhyp: bool = True,
    *,
    hermitian: bool = False,
) -> list[cvxpy.Constraint]:
    r"""Return CVX constraints for matrix geometric-mean hypograph cone [@fawzi2015matrixgeometric].

    The set of matrices that satisfy the constraints are `A`, `B`, `T` triples such
    that

    \[
    G_t(A, B) \geq T
    \]

    where `G_t(A, B)` is the matrix geometric mean function.

    Args:
        A: A cvxpy expression representing a matrix.
        B: A cvxpy expression representing a matrix.
        T: A cvxpy expression representing a matrix.
        t: The weight in the range `[0, 1]`.
        fullhyp: Whether to use the full hypograph or the restricted hypograph.
        hermitian: Whether the matrices are Hermitian or symmetric.

    Raises:
        ValueError: If the weight is not in the range `[0, 1]`.
        ValueError: If the matrices are not the same size.
        ValueError: If the matrices are not 2D or not square.

    Returns:
        A list of CVX constraints.

    """
    if t < 0 or t > 1:
        raise ValueError("The weight must be in the range [0, 1].")

    if A.shape != B.shape or B.shape != T.shape:
        raise ValueError("The matrices must be the same size.")
    _require_square_2d(A, "The matrices")

    if fullhyp:
        dim = A.shape[0]
        W = _symmetric_like_variable(dim, hermitian=hermitian)
        hypo_w = _geometric_mean_cone_recursion(A, B, W, float(t), hermitian=hermitian)
        return [*hypo_w, W >> T]
    return _geometric_mean_cone_recursion(A, B, T, float(t), hermitian=hermitian)


def _geometric_mean_cone_recursion(
    A: cvxpy.Expression,
    B: cvxpy.Expression,
    T: cvxpy.Expression,
    t: float,
    *,
    hermitian: bool = False,
) -> list[cvxpy.Constraint]:
    r"""Recursively constructs the matrix geometric mean cone constraints.

    Applicable when the full hypograph is not required [@fawzi2015matrixgeometric].

    The base case is when t = 0, 1, or 1/2.

    When t = 0, the constraints are that A and B are positive semidefinite and A = T.
    When t = 1, the constraints are that A and B are positive semidefinite and B = T.

    When t = 1/2, the constraints are that A and B are positive semidefinite and
    the block matrix

    \[
        \begin{bmatrix} A & T \\ T & B \end{bmatrix} \succeq 0
    \]

    (positive semidefinite).

    The other branches add constraints and then recursively call the function
    with an updated weight until a base case is reached.

    If no branch is taken, it uses the identity
    \[
        G_t(A, B) = G_{1-t}(B, A).
    \]

    Args:
        A: A cvxpy expression representing a matrix.
        B: A cvxpy expression representing a matrix.
        T: A cvxpy expression representing a matrix.
        t: The weight in the range `[0, 1]`.
        hermitian: Whether the matrices are Hermitian or symmetric.

    Returns:
        A list of CVX constraints.

    """
    dim = A.shape[0]
    p, q = _reduced_fraction_pq(t)

    if p == 0:
        return [A >> 0, B >> 0, A == T]
    if p == q:
        return [A >> 0, B >> 0, B == T]
    if 2 * p == q:
        return [cvxpy.bmat([[A, T], [T, B]]) >> 0]

    if _is_power_of_two(q):
        Z = _symmetric_like_variable(dim, hermitian=hermitian)
        if t < 1 / 2:
            return [
                cvxpy.bmat([[A, T], [T, Z]]) >> 0,
                *_geometric_mean_cone_recursion(A, B, Z, 2 * t, hermitian=hermitian),
            ]
        return [
            cvxpy.bmat([[B, T], [T, Z]]) >> 0,
            *_geometric_mean_cone_recursion(A, B, Z, 2 * t - 1, hermitian=hermitian),
        ]

    if _is_power_of_two(p) and t > 1 / 2:
        Z = _symmetric_like_variable(dim, hermitian=hermitian)
        t_inner = float((2 * p - q) / p)
        rec = _geometric_mean_cone_recursion(A, T, Z, t_inner, hermitian=hermitian)
        return [*rec, cvxpy.bmat([[Z, T], [T, B]]) >> 0]

    if t < 1 / 2:
        log2_floor = int(np.floor(np.log2(q)))
        X = _symmetric_like_variable(dim, hermitian=hermitian)
        t1 = float(p / (2**log2_floor))
        t2 = float((2**log2_floor) / q)
        return _geometric_mean_cone_recursion(A, B, X, t1, hermitian=hermitian) + _geometric_mean_cone_recursion(
            A, X, T, t2, hermitian=hermitian
        )

    return _geometric_mean_cone_recursion(B, A, T, 1 - t, hermitian=hermitian)
