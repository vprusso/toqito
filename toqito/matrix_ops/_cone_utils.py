"""Shared internal helpers for matrix geometric-mean cone constraints."""

from fractions import Fraction
from typing import Any

import cvxpy


def _require_2d(mat: Any, name: str) -> None:
    r"""Raise ``ValueError`` unless ``mat`` is two-dimensional (``ndim == 2``)."""
    if mat.ndim != 2:
        raise ValueError(f"{name} must be 2D.")


def _require_square_2d(mat: Any, name: str) -> None:
    r"""Raise ``ValueError`` unless ``mat`` is a square two-dimensional matrix."""
    _require_2d(mat, name)
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"{name} must be square.")


def _is_power_of_two(n: int) -> bool:
    r"""Check if an integer is a power of two.

    Args:
        n: The integer to check.

    Returns:
        True if ``n`` is a power of two, and ``False`` otherwise.

    """
    if n <= 0:
        return False
    return (n & (n - 1)) == 0


def _reduced_fraction_pq(t: float) -> tuple[int, int]:
    r"""Return coprime ``(p, q)`` with ``t`` close to ``p/q``.

    Args:
        t: Weight represented as a float.

    Returns:
        Numerator ``p`` and denominator ``q`` of the reduced approximation to ``t``.

    """
    r = Fraction(t).limit_denominator()
    return r.numerator, r.denominator


def _symmetric_like_variable(dim: int, *, hermitian: bool) -> cvxpy.Variable:
    r"""Create a symmetric or Hermitian CVXPY variable of shape ``(dim, dim)``.

    Args:
        dim: The matrix dimension.
        hermitian: Whether to create a Hermitian (otherwise symmetric) variable.

    Returns:
        A CVXPY matrix variable with matching symmetry type.

    """
    if hermitian:
        return cvxpy.Variable((dim, dim), hermitian=True)
    return cvxpy.Variable((dim, dim), symmetric=True)
