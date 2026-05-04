"""CVXPY constraints for the matrix geometric mean hypograph cone."""

from fractions import Fraction

import cvxpy
import numpy as np


def _is_power_of_two(n: int) -> bool:
    r"""Check if an integer is a power of two.

    Args:
        n: The integer to check.

    Returns:
        True if n is a power of two, False otherwise.

    """
    if n <= 0:
        return False
    return (n & (n - 1)) == 0


def _reduced_fraction_pq(t: float) -> tuple[int, int]:
    r"""Return coprime ``(p, q)`` with ``t`` close to ``p/q`` (lowest terms via ``limit_denominator``).

    Args:
        t: Weight in ``[0, 1]`` (float).

    Returns:
        Numerator ``p`` and denominator ``q`` of the reduced approximation to ``t``.

    """
    r = Fraction(t).limit_denominator()
    return r.numerator, r.denominator


def _symmetric_like_variable(dim: int, *, hermitian: bool) -> cvxpy.Variable:
    r"""Create a symmetric or hermitian cvxpy variable of size dim.

    Args:
        dim: The dimension of the variable.
        hermitian: Whether the variable should be Hermitian or symmetric.

    Returns:
        A cvxpy variable.

    """
    if hermitian:
        return cvxpy.Variable((dim, dim), hermitian=True)
    return cvxpy.Variable((dim, dim), symmetric=True)


def _matrix_geo_mean_cone_recursion(
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

    When t = 0, the constraints are the A and B are positive semidefinite and A = T.
    When t = 1, the constraints are the A and B are positive semidefinite and B = T.

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
    G_t(A, B) = G_t_2(B, A) where t_2 = 1 - t.
    \]

    Args:
        A: A cvxpy expression representing a matrix.
        B: A cvxpy expression representing a matrix.
        T: A cvxpy expression representing a matrix.
        t: The weight in the range [0, 1].
        hermitian: Whether the matrices are Hermitian or symmetric.

    Returns:
        A list of CVX constraints.

    """
    dim = int(A.shape[0])
    p, q = _reduced_fraction_pq(t)

    if t == 0:
        return [A >> 0, B >> 0, A == T]
    if t == 1:
        return [A >> 0, B >> 0, B == T]
    if t == 1 / 2:
        return [cvxpy.bmat([[A, T], [T, B]]) >> 0]

    if _is_power_of_two(q):
        Z = _symmetric_like_variable(dim, hermitian=hermitian)
        if t < 1 / 2:
            return [
                cvxpy.bmat([[A, T], [T, Z]]) >> 0,
                *_matrix_geo_mean_cone_recursion(A, B, Z, 2 * t, hermitian=hermitian),
            ]
        return [
            cvxpy.bmat([[B, T], [T, Z]]) >> 0,
            *_matrix_geo_mean_cone_recursion(A, B, Z, 2 * t - 1, hermitian=hermitian),
        ]

    if _is_power_of_two(p) and t > 1 / 2:
        Z = _symmetric_like_variable(dim, hermitian=hermitian)
        t_inner = float(Fraction(2 * p - q, p))
        rec = _matrix_geo_mean_cone_recursion(A, T, Z, t_inner, hermitian=hermitian)
        return [*rec, cvxpy.bmat([[Z, T], [T, B]]) >> 0]

    if t < 1 / 2:
        log2_floor = int(np.floor(np.log2(q)))
        X = _symmetric_like_variable(dim, hermitian=hermitian)
        t1 = float(Fraction(p) / Fraction(2**log2_floor))
        t2 = float(Fraction(2**log2_floor) / Fraction(q))
        return _matrix_geo_mean_cone_recursion(
            A, B, X, t1, hermitian=hermitian
        ) + _matrix_geo_mean_cone_recursion(A, X, T, t2, hermitian=hermitian)

    return _matrix_geo_mean_cone_recursion(B, A, T, 1 - t, hermitian=hermitian)


def matrix_geo_mean_hypo_cone(
    A: cvxpy.Expression,
    B: cvxpy.Expression,
    T: cvxpy.Expression,
    t: float = 1 / 2,
    fullhyp: bool = True,
    *,
    hermitian: bool = False,
) -> list[cvxpy.Constraint]:
    r"""Return CVX constraints for matrix geo-mean hypograph cone [@fawzi2015matrixgeometric].

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
        t: The weight in the range [0, 1].
        fullhyp: Whether to use the full hypograph or the restricted hypograph.
        hermitian: Whether the matrices are Hermitian or symmetric.

    Raises:
        ValueError: If the weight is not in the range [0, 1].
        ValueError: If the matrices are not the same size.
        ValueError: If the matrices are not square.

    Returns:
        A list of CVX constraints.

    """
    if t < 0 or t > 1:
        raise ValueError("The weight must be in the range [0, 1].")

    if A.shape != B.shape or B.shape != T.shape:
        raise ValueError("The matrices must be the same size.")
    if int(A.shape[0]) != int(A.shape[1]):
        raise ValueError("The matrices must be square.")

    if fullhyp:
        dim = int(A.shape[0])
        W = _symmetric_like_variable(dim, hermitian=hermitian)
        hypo_w = _matrix_geo_mean_cone_recursion(A, B, W, float(t), hermitian=hermitian)
        return [*hypo_w, W >> T]
    return _matrix_geo_mean_cone_recursion(A, B, T, float(t), hermitian=hermitian)
