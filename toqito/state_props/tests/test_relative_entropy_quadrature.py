"""Tests for relative_entropy_quadrature (numeric / constant only)."""

import re

import cvxpy
import numpy as np
import pytest

from toqito.cones._utils import _AFFINE_VARIABLE_USE_CONE
from toqito.state_props.relative_entropy_quadrature import relative_entropy_quadrature

_NOT_SUPPORTED = re.escape(_AFFINE_VARIABLE_USE_CONE)


def _rand_positive(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.1, 1.0, size=n)


def test_relative_entropy_quadrature_numeric():
    """Element-wise values match ``x * log(x / y)``."""
    vec_x = np.array([0.3, 0.7])
    vec_y = np.array([0.5, 0.5])
    expected = vec_x * np.log(vec_x / vec_y)
    np.testing.assert_allclose(relative_entropy_quadrature(vec_x, vec_y), expected, rtol=1e-12)


def test_relative_entropy_quadrature_rejects_nonconstant_affine():
    """Non-constant affine expressions are rejected."""
    vec_x = np.array([0.3, 0.7])
    w_var = cvxpy.Variable(2)
    w_var.value = np.zeros(2)
    expr = cvxpy.Constant(vec_x) + w_var - w_var
    with pytest.raises(ValueError, match=_NOT_SUPPORTED):
        relative_entropy_quadrature(expr, np.array([0.5, 0.5]))


def test_relative_entropy_quadrature_broadcast_scalar_x():
    """Scalar ``vec_x`` broadcasts like CVXQUAD."""
    vec_x = np.array([0.4])
    vec_y = np.array([0.2, 0.8])
    expected = vec_x * np.log(vec_x / vec_y)
    np.testing.assert_allclose(relative_entropy_quadrature(vec_x, vec_y), expected, rtol=1e-12)


def test_relative_entropy_quadrature_broadcast_scalar_y():
    """Scalar ``vec_y`` broadcasts like CVXQUAD."""
    vec_x = np.array([0.2, 0.8])
    vec_y = np.array([0.4])
    expected = vec_x * np.log(vec_x / vec_y)
    np.testing.assert_allclose(relative_entropy_quadrature(vec_x, vec_y), expected, rtol=1e-12)


def test_relative_entropy_quadrature_constant_cvxpy():
    """Constant CVXPY expressions recurse to the numeric path."""
    vec_x = np.array([0.25, 0.75])
    vec_y = np.array([0.5, 0.5])
    expected = vec_x * np.log(vec_x / vec_y)
    np.testing.assert_allclose(
        relative_entropy_quadrature(cvxpy.Constant(vec_x), cvxpy.Constant(vec_y)),
        expected,
        rtol=1e-12,
    )


def test_relative_entropy_quadrature_constant_cvxpy_with_numpy():
    """Mixed constant CVXPY and numpy inputs promote ``vec_y`` to ``Constant``."""
    vec_x = np.array([0.25, 0.75])
    vec_y = np.array([0.5, 0.5])
    expected = vec_x * np.log(vec_x / vec_y)
    got = relative_entropy_quadrature(cvxpy.Constant(vec_x), vec_y)
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_relative_entropy_quadrature_numpy_with_constant_cvxpy():
    """Mixed numpy and constant CVXPY inputs recurse through ``_constant_value``."""
    vec_x = np.array([0.25, 0.75])
    vec_y = np.array([0.5, 0.5])
    expected = vec_x * np.log(vec_x / vec_y)
    got = relative_entropy_quadrature(vec_x, cvxpy.Constant(vec_y))
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_relative_entropy_quadrature_broadcast_scalar_x_cvxpy():
    """Scalar CVXPY ``vec_x`` skips numpy broadcast in ``_broadcast_shape``."""
    vec_x = cvxpy.Constant(np.array([0.4]))
    vec_y = np.array([0.2, 0.8])
    expected = 0.4 * np.log(0.4 / vec_y)
    np.testing.assert_allclose(
        relative_entropy_quadrature(vec_x, vec_y),
        expected,
        rtol=1e-12,
    )


def test_relative_entropy_quadrature_broadcast_scalar_y_cvxpy():
    """Scalar CVXPY ``vec_y`` skips numpy broadcast in ``_broadcast_shape``."""
    vec_x = np.array([0.2, 0.8])
    vec_y = cvxpy.Constant(np.array([0.4]))
    expected = vec_x * np.log(vec_x / 0.4)
    np.testing.assert_allclose(
        relative_entropy_quadrature(vec_x, vec_y),
        expected,
        rtol=1e-12,
    )


def test_relative_entropy_quadrature_rejects_bare_variable():
    """Bare ``Variable`` inputs are not supported."""
    x_var = cvxpy.Variable(2)
    x_var.value = np.array([0.3, 0.7])
    with pytest.raises(ValueError, match=_NOT_SUPPORTED):
        relative_entropy_quadrature(x_var, np.array([0.5, 0.5]))


@pytest.mark.parametrize("n", [3, 5, 8])
def test_relative_entropy_quadrature_numeric_grid(n: int) -> None:
    """Numeric inputs match the element-wise reference."""
    vec_x = _rand_positive(n, seed=n * 100_003)
    vec_y = _rand_positive(n, seed=n * 100_003 + 1)
    ref_vec = vec_x * np.log(vec_x / vec_y)
    np.testing.assert_allclose(
        relative_entropy_quadrature(vec_x, vec_y),
        ref_vec,
        rtol=1e-10,
        atol=1e-10,
    )


def test_relative_entropy_quadrature_vec_x_wrong_type() -> None:
    """Reject ``vec_x`` that is not a numpy array or CVXPY expression."""
    with pytest.raises(
        ValueError,
        match=re.escape("vec_x must be a numpy array or a cvxpy expression"),
    ):
        relative_entropy_quadrature([0.3, 0.7], np.array([0.5, 0.5]))


def test_relative_entropy_quadrature_vec_y_wrong_type() -> None:
    """Reject ``vec_y`` that is not a numpy array or CVXPY expression."""
    with pytest.raises(
        ValueError,
        match=re.escape("vec_y must be a numpy array or a cvxpy expression"),
    ):
        relative_entropy_quadrature(np.array([0.3, 0.7]), [0.5, 0.5])


def test_relative_entropy_quadrature_incompatible_shapes() -> None:
    """Reject non-broadcastable ``vec_x`` and ``vec_y`` shapes."""
    with pytest.raises(
        ValueError,
        match=re.escape("The dimensions of vec_x and vec_y are not compatible."),
    ):
        relative_entropy_quadrature(np.array([0.3, 0.7]), np.array([0.5, 0.5, 0.6]))


def test_relative_entropy_quadrature_non_positive_numeric() -> None:
    """Reject non-positive numeric inputs."""
    with pytest.raises(ValueError, match=re.escape("vec_x and vec_y must be positive")):
        relative_entropy_quadrature(np.array([0.3, -0.1]), np.array([0.5, 0.5]))


def test_relative_entropy_quadrature_constant_no_value() -> None:
    """Reject constant CVXPY expressions with no ``.value``."""
    p = cvxpy.Parameter(2)
    assert p.value is None
    with pytest.raises(
        ValueError,
        match=re.escape("Constant CVXPY expression has no numeric value; set `.value` or pass a numpy.ndarray."),
    ):
        relative_entropy_quadrature(p, cvxpy.Constant(np.array([0.5, 0.5])))


def test_relative_entropy_quadrature_rejects_nonconstant_variable() -> None:
    """Reject non-constant CVXPY inputs."""
    x_var = cvxpy.Variable(2, nonneg=True)
    y_c = cvxpy.Constant(np.array([0.5, 0.5]))
    with pytest.raises(ValueError, match=_NOT_SUPPORTED):
        relative_entropy_quadrature(x_var, y_c)


def test_relative_entropy_quadrature_rejects_nonconstant_quadratic() -> None:
    """Reject non-constant quadratic CVXPY inputs."""
    x_var = cvxpy.Variable(2, nonneg=True)
    x_var.value = np.array([0.3, 0.7])
    y_c = cvxpy.Constant(np.array([0.5, 0.5]))
    with pytest.raises(ValueError, match=_NOT_SUPPORTED):
        relative_entropy_quadrature(cvxpy.square(x_var), y_c)
