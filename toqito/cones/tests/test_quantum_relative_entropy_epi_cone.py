"""Tests for quantum_relative_entropy_epi_cone."""

# Adapted from CVXQUAD (https://github.com/hfawzi/cvxquad), BSD-2-Clause.
# Original implementation by Fawzi, Saunderson, et al.

import re

import cvxpy
import numpy as np
import pytest

from toqito.cones.quantum_relative_entropy_epi_cone import (
    quantum_relative_entropy_epi_cone,
)
from toqito.matrix_props import is_positive_semidefinite
from toqito.state_props.quantum_relative_entropy import quantum_relative_entropy


def _rand_psd_normalized(dim: int, seed: int, *, hermitian: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if hermitian:
        g = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    else:
        g = rng.standard_normal((dim, dim))
    mat = g @ g.conj().T + 1e-1 * np.eye(dim, dtype=g.dtype)
    mat = (mat + mat.conj().T) / 2
    return mat / np.trace(mat)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("mk", [1, 3])
@pytest.mark.parametrize("apx", [-1, 0, 1])
@pytest.mark.parametrize("hermitian", [False, True])
def test_quantum_relative_entropy_epi_cone_at_constant(dim: int, mk: int, apx: int, hermitian: bool):
    """Minimize ``t`` at fixed Constants and compare to numeric QRE."""
    if mk == 1 and apx == 0:
        pytest.skip("CVXQUAD skips (m,k)=(1,1) with Pade apx=0.")

    seed = dim * 100_003 + mk * 17 + (apx + 1) * 3 + int(hermitian)
    mat_a = _rand_psd_normalized(dim, seed, hermitian=hermitian)
    mat_b = _rand_psd_normalized(dim, seed + 1, hermitian=hermitian)
    assert is_positive_semidefinite(np.asarray(mat_a, dtype=np.complex128))
    assert is_positive_semidefinite(np.asarray(mat_b, dtype=np.complex128))

    dab = quantum_relative_entropy(mat_a, mat_b)

    t = cvxpy.Variable()
    cons = quantum_relative_entropy_epi_cone(
        cvxpy.Constant(mat_a),
        cvxpy.Constant(mat_b),
        t,
        m=mk,
        k=mk,
        apx=apx,
        hermitian=hermitian,
    )
    prob = cvxpy.Problem(cvxpy.Minimize(t), cons)
    val = prob.solve(solver=cvxpy.SCS, verbose=False)

    assert prob.status in {cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE}, prob.status
    assert val is not None

    if abs(dab) < 1e-12:
        assert abs(val - dab) <= 1e-4
        return

    err = (val - dab) / abs(dab)
    if apx != 0:
        assert apx * err >= -5e-4, err
    if mk >= 3:
        assert abs(err) <= 1e-2, err


def test_quantum_relative_entropy_epi_cone_pure_vs_maximally_mixed():
    """``D(|0><0| || I/2)`` recovers ``ln(2)``."""
    rho = np.array([[1.0, 0.0], [0.0, 0.0]])
    sigma = np.eye(2) / 2
    ref = quantum_relative_entropy(rho, sigma)
    assert ref == pytest.approx(float(np.log(2)), abs=1e-12)

    t = cvxpy.Variable()
    cons = quantum_relative_entropy_epi_cone(
        cvxpy.Constant(rho),
        cvxpy.Constant(sigma),
        t,
        m=3,
        k=3,
        apx=0,
    )
    prob = cvxpy.Problem(cvxpy.Minimize(t), cons)
    val = prob.solve(solver=cvxpy.SCS, verbose=False)
    assert prob.status in {cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE}, prob.status
    assert val == pytest.approx(float(np.log(2)), abs=2e-2)


def test_quantum_relative_entropy_epi_cone_composition():
    """Free density matrix ``X``; minimize ``D(X || I/n)`` recovers zero at ``I/n``."""
    n = 2
    sigma = np.eye(n) / n
    x_var = cvxpy.Variable((n, n), symmetric=True)
    t = cvxpy.Variable()
    cons = quantum_relative_entropy_epi_cone(
        x_var,
        cvxpy.Constant(sigma),
        t,
        m=3,
        k=3,
        apx=0,
        hermitian=False,
    )
    cons.extend([x_var >> 0, cvxpy.trace(x_var) == 1])
    prob = cvxpy.Problem(cvxpy.Minimize(t), cons)
    val = prob.solve(solver=cvxpy.SCS, verbose=False)

    assert prob.status in {cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE}, prob.status
    assert val == pytest.approx(0.0, abs=5e-2)
    assert x_var.value is not None
    np.testing.assert_allclose(x_var.value, sigma, atol=5e-2)


def test_quantum_relative_entropy_epi_cone_mat_x_not_square() -> None:
    """Reject non-square ``mat_x``."""
    mat_x = cvxpy.Variable((2, 3))
    mat_y = cvxpy.Variable((2, 2), symmetric=True)
    t = cvxpy.Variable()
    with pytest.raises(ValueError, match=re.escape("mat_x must be square.")):
        quantum_relative_entropy_epi_cone(mat_x, mat_y, t)


def test_quantum_relative_entropy_epi_cone_shape_mismatch() -> None:
    """Reject mismatched shapes."""
    mat_x = cvxpy.Variable((2, 2), symmetric=True)
    mat_y = cvxpy.Variable((3, 3), symmetric=True)
    t = cvxpy.Variable()
    with pytest.raises(ValueError, match=re.escape("mat_x and mat_y must have the same shape")):
        quantum_relative_entropy_epi_cone(mat_x, mat_y, t)


def test_quantum_relative_entropy_epi_cone_m_invalid() -> None:
    """Reject non-positive ``m``."""
    mat_x = cvxpy.Variable((2, 2), symmetric=True)
    mat_y = cvxpy.Variable((2, 2), symmetric=True)
    t = cvxpy.Variable()
    with pytest.raises(ValueError, match=re.escape("m must be at least 1")):
        quantum_relative_entropy_epi_cone(mat_x, mat_y, t, m=0)


def test_quantum_relative_entropy_epi_cone_k_invalid() -> None:
    """Reject non-positive ``k``."""
    mat_x = cvxpy.Variable((2, 2), symmetric=True)
    mat_y = cvxpy.Variable((2, 2), symmetric=True)
    t = cvxpy.Variable()
    with pytest.raises(ValueError, match=re.escape("k must be at least 1")):
        quantum_relative_entropy_epi_cone(mat_x, mat_y, t, k=0)


def test_quantum_relative_entropy_epi_cone_apx_invalid() -> None:
    """Reject ``apx`` outside ``{-1, 0, 1}``."""
    mat_x = cvxpy.Variable((2, 2), symmetric=True)
    mat_y = cvxpy.Variable((2, 2), symmetric=True)
    t = cvxpy.Variable()
    with pytest.raises(ValueError, match=re.escape("apx must be -1, 0, or 1")):
        quantum_relative_entropy_epi_cone(mat_x, mat_y, t, apx=2)
