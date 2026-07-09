"""Shared internal helpers for matrix geometric-mean cone constraints."""

from fractions import Fraction
from typing import Any

import cvxpy
import numpy as np


def _contains_effective_variables(expr: cvxpy.Expression) -> bool:
    """Check whether CVXPY variables affect the expression value.

    Returns ``False`` for numpy arrays and constant expressions.  Returns
    ``True`` when at least one CVXPY variable syntactically present in *expr*
    actually changes the expression value when its numeric value is perturbed.

    This correctly handles cases like ``A + X - X`` where a variable is
    present syntactically but cancels out algebraically — that expression
    returns ``False``.

    Args:
        expr: The CVXPY expression to probe.

    Returns:
        ``True`` if the expression value is sensitive to at least one of its
        free variables; ``False`` otherwise.

    """
    if not isinstance(expr, cvxpy.Expression):
        return False
    if expr.is_constant():
        return False
    variables = expr.variables()
    if not variables:
        return False

    saved = [(v, v.value) for v in variables]

    try:
        try:
            val_before = expr.value
        except Exception:
            return True
        if val_before is None:
            return True

        rng = np.random.default_rng(0)
        for var in variables:
            old = var.value
            shape = var.shape
            perturbation = rng.standard_normal(shape) * 1e-4
            if var.is_symmetric():
                perturbation = (perturbation + perturbation.T) / 2
            elif var.is_hermitian():
                perturbation = (perturbation + perturbation.conj().T) / 2
            var.value = (old if old is not None else np.zeros(shape)) + perturbation

        try:
            val_after = expr.value
        except Exception:
            return True

        if val_after is None:
            return True

        return not np.allclose(val_before, val_after, rtol=1e-8, atol=1e-10)

    finally:
        for var, original in saved:
            var.value = original


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
