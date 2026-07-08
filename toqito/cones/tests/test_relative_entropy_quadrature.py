"""Tests for relative_entropy_quadrature."""

import re

import cvxpy
import numpy as np
import pytest

from toqito.cones.relative_entropy_quadrature import relative_entropy_quadrature


def _rand_positive(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.1, 1.0, size=n)


def _affine_fixed_at(vec: np.ndarray) -> cvxpy.Expression:
    """``Constant(v) + w - w`` with ``w.value = 0`` (algebraically ``v``)."""
    w_var = cvxpy.Variable(vec.shape)
    w_var.value = np.zeros(vec.shape)
    return cvxpy.Constant(vec) + w_var - w_var


def test_relative_entropy_quadrature_numeric():
    """Element-wise values match ``x * log(x / y)``."""
    vec_x = np.array([0.3, 0.7])
    vec_y = np.array([0.5, 0.5])
    expected = vec_x * np.log(vec_x / vec_y)
    np.testing.assert_allclose(relative_entropy_quadrature(vec_x, vec_y), expected, rtol=1e-12)


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
    x_c = cvxpy.Constant(vec_x)
    y_c = cvxpy.Constant(vec_y)
    expected = vec_x * np.log(vec_x / vec_y)
    np.testing.assert_allclose(relative_entropy_quadrature(x_c, y_c), expected, rtol=1e-12)


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


def test_relative_entropy_quadrature_numpy_with_affine():
    """A numpy input with an affine CVXPY partner promotes to ``Constant``."""
    vec_x = np.array([0.25, 0.75])
    vec_y = np.array([0.5, 0.5])
    ref_sum = float(np.sum(vec_x * np.log(vec_x / vec_y)))
    val = relative_entropy_quadrature(vec_x, _affine_fixed_at(vec_y), m=3, k=3)
    assert val == pytest.approx(ref_sum, rel=0, abs=1e-2)


def test_relative_entropy_quadrature_affine_with_numpy():
    """An affine CVXPY input with a numpy partner promotes to ``Constant``."""
    vec_x = np.array([0.25, 0.75])
    vec_y = np.array([0.5, 0.5])
    ref_sum = float(np.sum(vec_x * np.log(vec_x / vec_y)))
    val = relative_entropy_quadrature(_affine_fixed_at(vec_x), vec_y, m=3, k=3)
    assert val == pytest.approx(ref_sum, rel=0, abs=1e-2)


def test_relative_entropy_quadrature_sdp_failure(monkeypatch):
    """A failed SDP solve should raise ``ValueError``."""

    class FakeProblem:
        def __init__(self, objective, constraints):
            self.value = 1.0
            self.status = cvxpy.INFEASIBLE

        def solve(self, **kwargs):
            pass

    monkeypatch.setattr(cvxpy, "Problem", FakeProblem)

    vec_x = np.array([0.3, 0.7])
    vec_y = np.array([0.5, 0.5])
    with pytest.raises(ValueError, match=re.escape("The SDP did not solve successfully")):
        relative_entropy_quadrature(_affine_fixed_at(vec_x), _affine_fixed_at(vec_y), m=3, k=3)


@pytest.mark.parametrize("n", [3, 5, 8])
@pytest.mark.parametrize("mk", [1, 3])
@pytest.mark.parametrize(
    ("x_affine", "y_affine"),
    [
        (True, True),
        (True, False),
        (False, True),
    ],
    ids=["both_affine", "x_affine_y_constant", "x_constant_y_affine"],
)
def test_relative_entropy_quadrature_grid(
    n: int,
    mk: int,
    x_affine: bool,
    y_affine: bool,
) -> None:
    """Numeric and affine SDP accuracy over a random positive grid."""
    seed = n * 100_003 + mk * 17 + int(x_affine) * 2 + int(y_affine)
    vec_x = _rand_positive(n, seed=seed)
    vec_y = _rand_positive(n, seed=seed + 1)

    ref_vec = vec_x * np.log(vec_x / vec_y)
    ref_sum = float(np.sum(ref_vec))

    np.testing.assert_allclose(
        relative_entropy_quadrature(vec_x, vec_y),
        ref_vec,
        rtol=1e-10,
        atol=1e-10,
    )

    x_expr = _affine_fixed_at(vec_x) if x_affine else cvxpy.Constant(vec_x)
    y_expr = _affine_fixed_at(vec_y) if y_affine else cvxpy.Constant(vec_y)
    val = relative_entropy_quadrature(x_expr, y_expr, m=mk, k=mk)
    if mk == 1:
        assert abs(val - ref_sum) <= max(0.15, 0.12 * abs(ref_sum))
    else:
        assert val == pytest.approx(ref_sum, rel=0, abs=1e-2)


class TestRelativeEntropyQuadratureValueErrors:
    """``ValueError`` paths in ``relative_entropy_quadrature``."""

    def test_vec_x_wrong_type(self) -> None:
        """Reject ``vec_x`` that is not a numpy array or CVXPY expression."""
        with pytest.raises(
            ValueError,
            match=re.escape("vec_x must be a numpy array or a cvxpy expression"),
        ):
            relative_entropy_quadrature([0.3, 0.7], np.array([0.5, 0.5]))

    def test_vec_y_wrong_type(self) -> None:
        """Reject ``vec_y`` that is not a numpy array or CVXPY expression."""
        with pytest.raises(
            ValueError,
            match=re.escape("vec_y must be a numpy array or a cvxpy expression"),
        ):
            relative_entropy_quadrature(np.array([0.3, 0.7]), [0.5, 0.5])

    def test_m_less_than_one(self) -> None:
        """Reject quadrature count ``m`` below 1."""
        with pytest.raises(ValueError, match=re.escape("m must be at least 1")):
            relative_entropy_quadrature(np.array([0.3, 0.7]), np.array([0.5, 0.5]), m=0, k=3)

    def test_k_less_than_one(self) -> None:
        """Reject square-root count ``k`` below 1."""
        with pytest.raises(ValueError, match=re.escape("k must be at least 1")):
            relative_entropy_quadrature(np.array([0.3, 0.7]), np.array([0.5, 0.5]), m=3, k=0)

    def test_incompatible_shapes(self) -> None:
        """Reject non-broadcastable ``vec_x`` and ``vec_y`` shapes."""
        with pytest.raises(
            ValueError,
            match=re.escape("The dimensions of vec_x and vec_y are not compatible."),
        ):
            relative_entropy_quadrature(np.array([0.3, 0.7]), np.array([0.5, 0.5, 0.6]))

    def test_non_positive_numeric(self) -> None:
        """Reject non-positive numeric inputs."""
        with pytest.raises(ValueError, match=re.escape("vec_x and vec_y must be positive")):
            relative_entropy_quadrature(np.array([0.3, -0.1]), np.array([0.5, 0.5]))

    def test_constant_no_value(self) -> None:
        """Reject constant CVXPY expressions with no ``.value``."""
        p = cvxpy.Parameter(2)
        assert p.value is None
        with pytest.raises(
            ValueError,
            match=re.escape("Constant CVXPY expression has no numeric value; set `.value` or pass a numpy.ndarray."),
        ):
            relative_entropy_quadrature(p, cvxpy.Constant(np.array([0.5, 0.5])))

    def test_neither_affine(self) -> None:
        """Reject inputs when neither ``vec_x`` nor ``vec_y`` is affine."""
        x_var = cvxpy.Variable(2, nonneg=True)
        x_var.value = np.array([0.3, 0.7])
        y_var = cvxpy.Variable(2, nonneg=True)
        y_var.value = np.array([0.5, 0.5])
        with pytest.raises(
            ValueError,
            match=re.escape("At least one of vec_x and vec_y must be an affine CVXPY expression."),
        ):
            relative_entropy_quadrature(cvxpy.square(x_var), cvxpy.square(y_var))

    def test_affine_missing_initial_value(self) -> None:
        """Reject affine inputs with no initial ``.value`` for positivity checks."""
        x_var = cvxpy.Variable(2, nonneg=True)
        y_c = cvxpy.Constant(np.array([0.5, 0.5]))
        with pytest.raises(
            ValueError,
            match=re.escape("Affine vec_x and vec_y need numeric initial values; set `.value`."),
        ):
            relative_entropy_quadrature(x_var, y_c)

    def test_non_positive_at_initial_value(self) -> None:
        """Reject affine inputs that are not positive at ``.value``."""
        x_var = cvxpy.Variable(2)
        x_var.value = np.array([0.3, -0.1])
        y_c = cvxpy.Constant(np.array([0.5, 0.5]))
        with pytest.raises(
            ValueError,
            match=re.escape("vec_x and vec_y must be positive at the initial value."),
        ):
            relative_entropy_quadrature(x_var, y_c)
