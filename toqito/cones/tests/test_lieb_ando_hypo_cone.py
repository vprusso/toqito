"""Tests for lieb_ando_hypo_cone."""

import re
from fractions import Fraction

import cvxpy
import numpy as np
import pytest
from scipy.linalg import fractional_matrix_power

from toqito.cones.lieb_ando_hypo_cone import lieb_ando_hypo_cone
from toqito.matrix_props import is_positive_semidefinite
from toqito.matrix_props.lieb_ando import lieb_ando

DIMS = (2,)
WEIGHTS = (0.5, 0.25, 2 / 3)
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


def _numeric_lieb_ando_reference(mat_a: np.ndarray, mat_b: np.ndarray, mat_k: np.ndarray, t: float) -> float:
    a_sym = (mat_a + mat_a.conj().T) / 2
    b_sym = (mat_b + mat_b.conj().T) / 2
    a_raised = fractional_matrix_power(a_sym, 1.0 - float(t))
    b_raised = fractional_matrix_power(b_sym, float(t))
    return float(np.real(np.trace(mat_k.conj().T @ a_raised @ mat_k @ b_raised)))


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("power", WEIGHTS)
@pytest.mark.parametrize("hermitian", [False, True])
def test_lieb_ando_hypo_cone_at_constant(dim: int, power: float, hermitian: bool):
    """Maximize ``t`` at fixed Constants and compare to the numeric Lieb--Ando value."""
    seed = _case_seed(dim, power, hermitian=hermitian)
    mat_a = _random_pd_matrix(dim, seed, hermitian=hermitian)
    mat_b = _random_pd_matrix(dim, seed + 1, hermitian=hermitian)
    mat_k = _random_pd_matrix(dim, seed + 2, hermitian=hermitian)
    assert is_positive_semidefinite(np.asarray(mat_a, dtype=np.complex128))
    assert is_positive_semidefinite(np.asarray(mat_b, dtype=np.complex128))

    ref = _numeric_lieb_ando_reference(mat_a, mat_b, mat_k, power)
    np.testing.assert_allclose(lieb_ando(mat_a, mat_b, mat_k, power), ref, rtol=1e-8, atol=1e-8)

    t = cvxpy.Variable()
    cons = lieb_ando_hypo_cone(
        cvxpy.Constant(mat_a),
        cvxpy.Constant(mat_b),
        mat_k,
        t,
        power,
        hermitian=hermitian,
    )
    prob = cvxpy.Problem(cvxpy.Maximize(t), cons)
    cvx_val = prob.solve(solver=cvxpy.SCS, verbose=False)

    assert prob.status in {cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE}, prob.status
    assert cvx_val is not None
    np.testing.assert_allclose(float(cvx_val), ref, rtol=2e-2, atol=1e-2)


def test_lieb_ando_hypo_cone_identity_k():
    """Identity ``K`` on a diagonal pair recovers a simple closed form."""
    mat_a = np.diag([4.0, 1.0])
    mat_b = np.diag([9.0, 4.0])
    mat_k = np.eye(2)
    power = 0.5
    ref = lieb_ando(mat_a, mat_b, mat_k, power)

    t = cvxpy.Variable()
    cons = lieb_ando_hypo_cone(
        cvxpy.Constant(mat_a),
        cvxpy.Constant(mat_b),
        mat_k,
        t,
        power,
    )
    prob = cvxpy.Problem(cvxpy.Maximize(t), cons)
    val = prob.solve(solver=cvxpy.SCS, verbose=False)
    assert prob.status in {cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE}, prob.status
    assert val == pytest.approx(ref, abs=2e-2)


def test_lieb_ando_hypo_cone_composition():
    """Free symmetric ``A`` with fixed ``B`` and ``K``; maximize Lieb--Ando."""
    n = 2
    eps = 0.25
    mat_b = np.eye(n)
    mat_k = np.eye(n)
    a_var = cvxpy.Variable((n, n), symmetric=True)
    t = cvxpy.Variable()
    cons = lieb_ando_hypo_cone(a_var, cvxpy.Constant(mat_b), mat_k, t, 0.5)
    cons.extend([a_var >> eps * np.eye(n), a_var << np.eye(n)])
    prob = cvxpy.Problem(cvxpy.Maximize(t), cons)
    val = prob.solve(solver=cvxpy.SCS, verbose=False)

    assert prob.status in {cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE}, prob.status
    # On this set, Lieb--Ando for K=I, B=I, t=1/2 is tr(A^{1/2}), max at A=I.
    assert val == pytest.approx(float(n), abs=5e-2)
    assert a_var.value is not None
    np.testing.assert_allclose(a_var.value, np.eye(n), atol=5e-2)


def test_lieb_ando_hypo_cone_mat_a_not_square() -> None:
    """Reject non-square ``mat_a``."""
    mat_a = cvxpy.Variable((2, 3))
    mat_b = cvxpy.Variable((3, 3), symmetric=True)
    t = cvxpy.Variable()
    with pytest.raises(ValueError, match=re.escape("mat_a must be square.")):
        lieb_ando_hypo_cone(mat_a, mat_b, np.ones((2, 3)), t, 0.5)


def test_lieb_ando_hypo_cone_mat_b_not_square() -> None:
    """Reject non-square ``mat_b``."""
    mat_a = cvxpy.Variable((2, 2), symmetric=True)
    mat_b = cvxpy.Variable((2, 3))
    t = cvxpy.Variable()
    with pytest.raises(ValueError, match=re.escape("mat_b must be square.")):
        lieb_ando_hypo_cone(mat_a, mat_b, np.ones((2, 2)), t, 0.5)


def test_lieb_ando_hypo_cone_power_invalid() -> None:
    """Reject ``power`` outside ``[0, 1]``."""
    mat_a = cvxpy.Variable((2, 2), symmetric=True)
    mat_b = cvxpy.Variable((2, 2), symmetric=True)
    t = cvxpy.Variable()
    with pytest.raises(ValueError, match=re.escape("power must be in the range [0, 1]")):
        lieb_ando_hypo_cone(mat_a, mat_b, np.eye(2), t, 1.5)


def test_lieb_ando_hypo_cone_mat_k_wrong_type() -> None:
    """Reject non-numpy ``mat_k``."""
    mat_a = cvxpy.Variable((2, 2), symmetric=True)
    mat_b = cvxpy.Variable((2, 2), symmetric=True)
    t = cvxpy.Variable()
    with pytest.raises(ValueError, match=re.escape("mat_k must be a numpy array")):
        lieb_ando_hypo_cone(mat_a, mat_b, cvxpy.Constant(np.eye(2)), t, 0.5)


def test_lieb_ando_hypo_cone_mat_k_shape_mismatch() -> None:
    """Reject incompatible ``mat_k`` shape."""
    mat_a = cvxpy.Variable((2, 2), symmetric=True)
    mat_b = cvxpy.Variable((2, 2), symmetric=True)
    t = cvxpy.Variable()
    with pytest.raises(
        ValueError,
        match=re.escape("mat_k must have the same number of rows as mat_a and the same number of columns as mat_b."),
    ):
        lieb_ando_hypo_cone(mat_a, mat_b, np.eye(3), t, 0.5)
