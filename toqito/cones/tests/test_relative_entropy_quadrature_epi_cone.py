"""Tests for relative_entropy_quadrature_epi_cone."""

import re

import cvxpy
import numpy as np
import pytest

from toqito.cones.relative_entropy_quadrature_epi_cone import (
    relative_entropy_quadrature_epi_cone,
)
from toqito.state_props.relative_entropy_quadrature import relative_entropy_quadrature


def _rand_positive(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.1, 1.0, size=n)


@pytest.mark.parametrize("n", [2, 3])
@pytest.mark.parametrize("mk", [3])
@pytest.mark.parametrize("apx", [0, -1, 1])
def test_relative_entropy_quadrature_epi_cone_at_constant(n: int, mk: int, apx: int):
    """Minimize ``sum(z)`` at fixed Constants and compare to ``x log(x/y)``."""
    vec_x = _rand_positive(n, seed=n * 100_003 + mk + (apx + 1))
    vec_y = _rand_positive(n, seed=n * 100_003 + mk + (apx + 1) + 1)
    ref = relative_entropy_quadrature(vec_x, vec_y)

    z = cvxpy.Variable(n)
    cons = relative_entropy_quadrature_epi_cone(
        cvxpy.Constant(vec_x),
        cvxpy.Constant(vec_y),
        z,
        m=mk,
        k=mk,
        apx=apx,
    )
    prob = cvxpy.Problem(cvxpy.Minimize(cvxpy.sum(z)), cons)
    val = prob.solve(solver=cvxpy.SCS, verbose=False)

    assert prob.status in {cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE}, prob.status
    assert val is not None
    np.testing.assert_allclose(float(val), float(np.sum(ref)), rtol=3e-2, atol=2e-2)
    assert z.value is not None
    np.testing.assert_allclose(np.asarray(z.value).ravel(), ref, rtol=3e-2, atol=2e-2)


def test_relative_entropy_quadrature_epi_cone_known_pair():
    """Sanity check on a fixed length-2 pair."""
    vec_x = np.array([0.3, 0.7])
    vec_y = np.array([0.5, 0.5])
    ref = relative_entropy_quadrature(vec_x, vec_y)
    z = cvxpy.Variable(2)
    cons = relative_entropy_quadrature_epi_cone(
        cvxpy.Constant(vec_x), cvxpy.Constant(vec_y), z
    )
    prob = cvxpy.Problem(cvxpy.Minimize(cvxpy.sum(z)), cons)
    val = prob.solve(solver=cvxpy.SCS, verbose=False)
    assert prob.status in {cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE}, prob.status
    assert val == pytest.approx(float(np.sum(ref)), abs=2e-2)


def test_relative_entropy_quadrature_epi_cone_broadcast_scalar_x():
    """Scalar ``vec_x`` broadcasts to ``vec_y``'s shape."""
    vec_y = np.array([0.2, 0.8])
    ref = 0.4 * np.log(0.4 / vec_y)
    z = cvxpy.Variable(2)
    cons = relative_entropy_quadrature_epi_cone(
        cvxpy.Constant(0.4),
        cvxpy.Constant(vec_y),
        z,
    )
    prob = cvxpy.Problem(cvxpy.Minimize(cvxpy.sum(z)), cons)
    val = prob.solve(solver=cvxpy.SCS, verbose=False)
    assert prob.status in {cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE}, prob.status
    assert val == pytest.approx(float(np.sum(ref)), abs=2e-2)
    assert z.value is not None
    np.testing.assert_allclose(np.asarray(z.value).ravel(), ref, rtol=3e-2, atol=2e-2)


def test_relative_entropy_quadrature_epi_cone_broadcast_scalar_y():
    """Scalar ``vec_y`` broadcasts to ``vec_x``'s shape."""
    vec_x = np.array([0.2, 0.8])
    ref = vec_x * np.log(vec_x / 0.4)
    z = cvxpy.Variable(2)
    cons = relative_entropy_quadrature_epi_cone(
        cvxpy.Constant(vec_x),
        cvxpy.Constant(0.4),
        z,
    )
    prob = cvxpy.Problem(cvxpy.Minimize(cvxpy.sum(z)), cons)
    val = prob.solve(solver=cvxpy.SCS, verbose=False)
    assert prob.status in {cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE}, prob.status
    assert val == pytest.approx(float(np.sum(ref)), abs=2e-2)


def test_relative_entropy_quadrature_epi_cone_both_scalars():
    """Both size-1 inputs skip broadcast expansion (``x_size == y_size == 1``)."""
    ref = 0.3 * np.log(0.3 / 0.5)
    z = cvxpy.Variable()
    cons = relative_entropy_quadrature_epi_cone(
        cvxpy.Constant(0.3),
        cvxpy.Constant(0.5),
        z,
    )
    prob = cvxpy.Problem(cvxpy.Minimize(z), cons)
    val = prob.solve(solver=cvxpy.SCS, verbose=False)
    assert prob.status in {cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE}, prob.status
    assert val == pytest.approx(ref, abs=2e-2)


def test_relative_entropy_quadrature_epi_cone_empty_vectors():
    """Empty broadcast shape returns no constraints."""
    z = cvxpy.Variable(0)
    cons = relative_entropy_quadrature_epi_cone(
        cvxpy.Constant(np.array([])),
        cvxpy.Constant(np.array([])),
        z,
    )
    assert cons == []


def test_relative_entropy_quadrature_epi_cone_composition():
    """Free positive ``x`` with fixed ``y``; minimize recovers ``x = y``."""
    n = 2
    vec_y = np.array([0.5, 0.5])
    x_var = cvxpy.Variable(n, pos=True)
    z = cvxpy.Variable(n)
    cons = relative_entropy_quadrature_epi_cone(x_var, cvxpy.Constant(vec_y), z)
    cons.append(cvxpy.sum(x_var) == 1)
    prob = cvxpy.Problem(cvxpy.Minimize(cvxpy.sum(z)), cons)
    val = prob.solve(solver=cvxpy.SCS, verbose=False)

    assert prob.status in {cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE}, prob.status
    assert val == pytest.approx(0.0, abs=5e-2)
    assert x_var.value is not None
    np.testing.assert_allclose(np.asarray(x_var.value).ravel(), vec_y, atol=5e-2)


def test_relative_entropy_quadrature_epi_cone_shape_mismatch() -> None:
    """Reject incompatible ``vec_x`` / ``vec_y`` shapes."""
    z = cvxpy.Variable(2)
    with pytest.raises(
        ValueError,
        match=re.escape("The dimensions of vec_x and vec_y are not compatible."),
    ):
        relative_entropy_quadrature_epi_cone(
            cvxpy.Variable(2),
            cvxpy.Variable(3),
            z,
        )


def test_relative_entropy_quadrature_epi_cone_z_shape_mismatch() -> None:
    """Reject ``z`` that does not match the broadcast shape."""
    with pytest.raises(
        ValueError,
        match=re.escape("z must have the broadcast shape of vec_x and vec_y"),
    ):
        relative_entropy_quadrature_epi_cone(
            cvxpy.Variable(2),
            cvxpy.Variable(2),
            cvxpy.Variable(3),
        )


def test_relative_entropy_quadrature_epi_cone_m_invalid() -> None:
    """Reject ``m`` below 1."""
    z = cvxpy.Variable(2)
    with pytest.raises(ValueError, match=re.escape("m must be at least 1")):
        relative_entropy_quadrature_epi_cone(
            cvxpy.Variable(2), cvxpy.Variable(2), z, m=0
        )


def test_relative_entropy_quadrature_epi_cone_k_invalid() -> None:
    """Reject ``k`` below 1."""
    z = cvxpy.Variable(2)
    with pytest.raises(ValueError, match=re.escape("k must be at least 1")):
        relative_entropy_quadrature_epi_cone(
            cvxpy.Variable(2), cvxpy.Variable(2), z, k=0
        )


def test_relative_entropy_quadrature_epi_cone_apx_invalid() -> None:
    """Reject invalid ``apx``."""
    z = cvxpy.Variable(2)
    with pytest.raises(ValueError, match=re.escape("apx must be -1, 0, or 1")):
        relative_entropy_quadrature_epi_cone(
            cvxpy.Variable(2), cvxpy.Variable(2), z, apx=2
        )
