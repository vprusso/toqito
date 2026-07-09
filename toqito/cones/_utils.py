"""Shared internal helpers for matrix geometric-mean cone constraints."""

from fractions import Fraction
from typing import Any

import cvxpy
import numpy as np


def _require_2d(mat: Any, name: str) -> None:
    r"""Raise ``ValueError`` unless ``mat`` is two-dimensional (``ndim == 2``)."""
    if mat.ndim != 2:
        raise ValueError(f"{name} must be 2D.")


def _require_square_2d(mat: Any, name: str) -> None:
    r"""Raise ``ValueError`` unless ``mat`` is a square two-dimensional matrix."""
    _require_2d(mat, name)
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"{name} must be square.")


def _contains_effective_variables(expr: cvxpy.Expression) -> bool:
    r"""Return whether CVXPY variables can change an affine expression's value."""
    if expr.is_constant() or not expr.variables():
        return False
    base_value = expr.value
    if base_value is None:
        return True
    base_value = np.asarray(base_value)
    saved_values = [(var, None if var.value is None else np.array(var.value, copy=True)) for var in expr.variables()]
    try:
        for probe_value in (0.0, 1.0):
            for var, _ in saved_values:
                var.value = probe_value if var.shape == () else probe_value * np.ones(var.shape)
            probed_value = expr.value
            if probed_value is None or not np.allclose(base_value, probed_value):
                return True
    except ValueError:
        return True
    finally:
        for var, saved_value in saved_values:
            var.value = saved_value
    return False


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
