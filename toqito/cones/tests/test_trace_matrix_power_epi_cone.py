"""Tests for trace_matrix_power_epi_cone."""

import re
from fractions import Fraction

import cvxpy
import numpy as np
import pytest
from scipy.linalg import fractional_matrix_power

from toqito.cones.trace_matrix_power_epi_cone import trace_matrix_power_epi_cone
from toqito.matrix_props import is_positive_semidefinite
from toqito.matrix_props.trace_matrix_power import trace_matrix_power

DIMS = (2, 3)
WEIGHTS = (-0.5, -0.25, 1.5, 1.75)
PD_SHIFT = 1e-1


def _case_seed(dim: int, t: float, *, hermitian: bool) -> int:
    r = Fraction(float(t)).limit_denominator()
    return int(dim * 1_000_003 + r.numerator * 10_009 + r.denominator * 100 + int(hermitian))


def _random_pd_matrix(dim: int, seed: int, *, hermitian: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if hermitian:
        x_mat = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    else:
        x_mat = rng.standard_normal((dim, dim))
    mat = x_mat @ x_mat.conj().T + PD_SHIFT * np.eye(dim, dtype=x_mat.dtype)
    return (mat + mat.conj().T) / 2


def _numeric_reference(mat_a: np.ndarray, power: float, mat_c: np.ndarray) -> float:
    a_sym = (mat_a + mat_a.conj().T) / 2
    powered = fractional_matrix_power(a_sym, float(power))
    return float(np.real(np.trace(mat_c @ powered)))


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("power", WEIGHTS)
@pytest.mark.parametrize("hermitian", [False, True])
def test_trace_matrix_power_epi_cone_at_constant(dim: int, power: float, hermitian: bool):
    """Minimize ``t`` at fixed ``Constant(A)`` and compare to ``tr(C A^p)``."""
    seed = _case_seed(dim, power, hermitian=hermitian)
    mat_a = _random_pd_matrix(dim, seed, hermitian=hermitian)
    mat_c = _random_pd_matrix(dim, seed + 2, hermitian=hermitian)
    assert is_positive_semidefinite(np.asarray(mat_a, dtype=np.complex128))
    assert is_positive_semidefinite(np.asarray(mat_c, dtype=np.complex128))

    ref = _numeric_reference(mat_a, power, mat_c)
    np.testing.assert_allclose(trace_matrix_power(mat_a, power, mat_c), ref, rtol=1e-8, atol=1e-8)

    t = cvxpy.Variable()
    cons = trace_matrix_power_epi_cone(
        cvxpy.Constant(mat_a),
        t,
        power,
        mat_c,
        hermitian=hermitian,
    )
    prob = cvxpy.Problem(cvxpy.Minimize(t), cons)
    cvx_val = prob.solve(solver=cvxpy.SCS, verbose=False)

    assert prob.status in {cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE}, prob.status
    assert cvx_val is not None
    np.testing.assert_allclose(float(cvx_val), ref, rtol=2.5e-2, atol=1e-2)


def test_trace_matrix_power_epi_cone_default_mat_c_is_identity():
    """Omitted ``mat_c`` uses the identity weight."""
    n = 2
    mat_a = np.eye(n)
    t = cvxpy.Variable()
    cons = trace_matrix_power_epi_cone(cvxpy.Constant(mat_a), t, 1.5)
    prob = cvxpy.Problem(cvxpy.Minimize(t), cons)
    val = prob.solve(solver=cvxpy.SCS, verbose=False)
    assert prob.status in {cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE}, prob.status
    assert val == pytest.approx(float(n), abs=1e-2)


def test_trace_matrix_power_epi_cone_composition():
    """Free ``Variable`` with ``eps I ⪯ A ⪯ I``; minimize ``tr(A^{3/2})``."""
    n = 2
    eps = 0.25
    a_var = cvxpy.Variable((n, n), symmetric=True)
    t = cvxpy.Variable()
    cons = trace_matrix_power_epi_cone(a_var, t, 1.5, hermitian=False)
    cons.extend([a_var >> eps * np.eye(n), a_var << np.eye(n)])
    prob = cvxpy.Problem(cvxpy.Minimize(t), cons)
    val = prob.solve(solver=cvxpy.SCS, verbose=False)

    assert prob.status in {cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE}, prob.status
    # Optimum of tr(A^{3/2}) on this set is at A = eps I.
    assert val == pytest.approx(n * (eps**1.5), abs=5e-2)
    assert a_var.value is not None
    np.testing.assert_allclose(a_var.value, eps * np.eye(n), atol=5e-2)


def test_trace_matrix_power_epi_cone_mat_a_not_square() -> None:
    """Reject non-square ``mat_a``."""
    mat_a = cvxpy.Variable((2, 3))
    t = cvxpy.Variable()
    with pytest.raises(ValueError, match=re.escape("mat_a must be square.")):
        trace_matrix_power_epi_cone(mat_a, t, 1.5)


def test_trace_matrix_power_epi_cone_power_invalid() -> None:
    """Reject ``power`` outside ``[-1, 0] U [1, 2]``."""
    mat_a = cvxpy.Variable((2, 2), symmetric=True)
    t = cvxpy.Variable()
    with pytest.raises(
        ValueError,
        match=re.escape("power must be in the range [-1, 0] or [1, 2]"),
    ):
        trace_matrix_power_epi_cone(mat_a, t, 0.5)


def test_trace_matrix_power_epi_cone_mat_c_wrong_type() -> None:
    """Reject non-numpy ``mat_c``."""
    mat_a = cvxpy.Variable((2, 2), symmetric=True)
    t = cvxpy.Variable()
    with pytest.raises(ValueError, match=re.escape("mat_c must be a numpy array")):
        trace_matrix_power_epi_cone(mat_a, t, 1.5, mat_c=cvxpy.Constant(np.eye(2)))


def test_trace_matrix_power_epi_cone_mat_c_not_psd() -> None:
    """Reject non-PSD ``mat_c``."""
    mat_a = cvxpy.Variable((2, 2), symmetric=True)
    t = cvxpy.Variable()
    with pytest.raises(
        ValueError,
        match=re.escape("mat_c must be a positive semidefinite matrix"),
    ):
        trace_matrix_power_epi_cone(mat_a, t, 1.5, mat_c=np.diag([1.0, -0.5]))


def test_trace_matrix_power_epi_cone_shape_mismatch() -> None:
    """Reject mismatched ``mat_a`` and ``mat_c`` shapes."""
    mat_a = cvxpy.Variable((2, 2), symmetric=True)
    t = cvxpy.Variable()
    with pytest.raises(
        ValueError,
        match=re.escape("mat_a and mat_c must have the same shape"),
    ):
        trace_matrix_power_epi_cone(mat_a, t, 1.5, mat_c=np.eye(3))
