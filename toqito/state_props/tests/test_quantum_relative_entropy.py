"""Tests for quantum_relative_entropy (numeric / constant only)."""

# Adapted from CVXQUAD (https://github.com/hfawzi/cvxquad), BSD-2-Clause.
# Original implementation by Fawzi, Saunderson, et al.

import re

import cvxpy
import numpy as np
import pytest
from scipy.linalg import logm

from toqito.cones._utils import _AFFINE_VARIABLE_USE_CONE
from toqito.state_props.quantum_relative_entropy import quantum_relative_entropy

_NOT_SUPPORTED = re.escape(_AFFINE_VARIABLE_USE_CONE)


def _rand_psd(dim: int, seed: int, *, hermitian: bool) -> np.ndarray:
    """Random PSD matrix."""
    rng = np.random.default_rng(seed)
    if hermitian:
        g = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    else:
        g = rng.standard_normal((dim, dim))
    mat = g @ g.conj().T + 1e-1 * np.eye(dim, dtype=g.dtype)
    return (mat + mat.conj().T) / 2


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("hermitian", [False, True])
def test_quantum_relative_entropy_numeric_grid(dim: int, hermitian: bool):
    """Numeric path is finite and nonnegative on random normalized PSD pairs."""
    seed = dim * 100_003 + int(hermitian)
    mat_a = _rand_psd(dim, seed, hermitian=hermitian)
    mat_b = _rand_psd(dim, seed + 1, hermitian=hermitian)
    mat_a = mat_a / np.trace(mat_a)
    mat_b = mat_b / np.trace(mat_b)
    val = quantum_relative_entropy(mat_a, mat_b)
    assert np.isfinite(val)
    assert val >= -1e-9


def test_quantum_relative_entropy_commuting_reference():
    """Diagonal ``X``, ``Y`` share eigenbasis; match ``tr(X(log X - log Y))``."""
    mat_x = np.diag([0.7, 0.3])
    mat_y = np.diag([0.4, 0.6])
    ref = float(np.real(np.trace(mat_x @ (logm(mat_x) - logm(mat_y)))))
    got = quantum_relative_entropy(mat_x, mat_y)
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-10)


def test_quantum_relative_entropy_constant_x_matches_numeric():
    """Constant ``mat_x`` unwraps to the numeric path."""
    n = 2
    rng = np.random.default_rng(21)
    g = rng.standard_normal((n, n))
    mat_a = g @ g.T + np.eye(n)
    mat_a = (mat_a + mat_a.T) / 2
    mat_a = mat_a / np.trace(mat_a)
    mat_b = np.eye(n) / n
    got = quantum_relative_entropy(cvxpy.Constant(mat_a), mat_b)
    want = quantum_relative_entropy(mat_a, mat_b)
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-10)


def test_quantum_relative_entropy_constant_y_matches_numeric():
    """Constant ``mat_y`` unwraps to the numeric path."""
    mat_a = np.diag([0.7, 0.3])
    mat_b = np.diag([0.4, 0.6])
    got = quantum_relative_entropy(mat_a, cvxpy.Constant(mat_b))
    want = quantum_relative_entropy(mat_a, mat_b)
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-10)


def test_quantum_relative_entropy_constant_x_constant_y_matches_numeric():
    """Both Constants unwrap to the numeric path."""
    n = 2
    rng = np.random.default_rng(27)
    g = rng.standard_normal((n, n))
    mat_a = g @ g.T + np.eye(n)
    mat_a = (mat_a + mat_a.T) / 2
    mat_a = mat_a / np.trace(mat_a)
    mat_b = np.eye(n) / n
    got = quantum_relative_entropy(cvxpy.Constant(mat_a), cvxpy.Constant(mat_b))
    want = quantum_relative_entropy(mat_a, mat_b)
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-10)


def test_quantum_relative_entropy_support_inclusion_failure():
    """Numeric path rejects ``im(X) not subseteq im(Y)``."""
    mat_x = np.array([[1.0, 0.0], [0.0, 0.0]])
    mat_y = np.array([[0.0, 0.0], [0.0, 1.0]])
    with pytest.raises(
        ValueError,
        match=re.escape("D(X||Y) is infinity because im(X) is not contained in im(Y)"),
    ):
        quantum_relative_entropy(mat_x, mat_y)


def test_quantum_relative_entropy_mat_x_wrong_type():
    """Reject ``mat_x`` that is not a numpy array or CVXPY expression."""
    with pytest.raises(
        ValueError,
        match=re.escape("mat_x must be a numpy array or a cvxpy expression"),
    ):
        quantum_relative_entropy([[1.0, 0.0], [0.0, 1.0]], np.eye(2))


def test_quantum_relative_entropy_mat_y_wrong_type():
    """Reject ``mat_y`` that is not a numpy array or CVXPY expression."""
    with pytest.raises(
        ValueError,
        match=re.escape("mat_y must be a numpy array or a cvxpy expression"),
    ):
        quantum_relative_entropy(np.eye(2), [[1.0, 0.0], [0.0, 1.0]])


def test_quantum_relative_entropy_shape_mismatch():
    """Reject ``mat_x`` and ``mat_y`` with different shapes."""
    with pytest.raises(
        ValueError,
        match=re.escape("mat_x and mat_y must have the same shape"),
    ):
        quantum_relative_entropy(np.eye(2), np.eye(3))


def test_quantum_relative_entropy_mat_x_not_2d():
    """Reject non-2D ``mat_x``."""
    with pytest.raises(ValueError, match=re.escape("mat_x must be 2D.")):
        quantum_relative_entropy(np.array([1.0, 0.0]), np.eye(2))


def test_quantum_relative_entropy_mat_y_not_square():
    """Reject non-square ``mat_y``."""
    with pytest.raises(ValueError, match=re.escape("mat_y must be square.")):
        quantum_relative_entropy(np.eye(2), np.zeros((2, 3)))


def test_quantum_relative_entropy_mat_x_not_psd():
    """Reject non-PSD numeric ``mat_x``."""
    mat_x = np.array([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(
        ValueError,
        match=re.escape("mat_x must be a positive semidefinite matrix"),
    ):
        quantum_relative_entropy(mat_x, np.eye(2))


def test_quantum_relative_entropy_mat_y_not_psd():
    """Reject non-PSD numeric ``mat_y``."""
    mat_y = np.diag([1.0, -0.5])
    with pytest.raises(
        ValueError,
        match=re.escape("mat_y must be a positive semidefinite matrix"),
    ):
        quantum_relative_entropy(np.eye(2), mat_y)


def test_quantum_relative_entropy_mat_x_not_hermitian():
    """Reject non-Hermitian numeric ``mat_x``."""
    mat_x = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.complex128)
    with pytest.raises(ValueError, match=re.escape("mat_x must be a Hermitian matrix")):
        quantum_relative_entropy(mat_x, np.eye(2))


def test_quantum_relative_entropy_mat_y_not_hermitian():
    """Reject non-Hermitian numeric ``mat_y``."""
    mat_y = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.complex128)
    with pytest.raises(ValueError, match=re.escape("mat_y must be a Hermitian matrix")):
        quantum_relative_entropy(np.eye(2), mat_y)


def test_quantum_relative_entropy_constant_x_no_value():
    """Reject constant ``mat_x`` with no ``.value``."""
    n = 2
    p_x = cvxpy.Parameter((n, n), symmetric=True)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Constant CVXPY expression has no numeric value; set parameter `.value` "
            "or pass mat_x as a numpy.ndarray."
        ),
    ):
        quantum_relative_entropy(p_x, np.eye(n))


def test_quantum_relative_entropy_constant_y_no_value():
    """Reject constant ``mat_y`` with no ``.value``."""
    n = 2
    p_y = cvxpy.Parameter((n, n), symmetric=True)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Constant CVXPY expression has no numeric value; set parameter `.value` "
            "or pass mat_y as a numpy.ndarray."
        ),
    ):
        quantum_relative_entropy(np.eye(n), p_y)


def test_quantum_relative_entropy_rejects_free_variable():
    """Free CVXPY Variables are rejected."""
    n = 2
    x_var = cvxpy.Variable((n, n), symmetric=True)
    x_var.value = np.eye(n)
    with pytest.raises(ValueError, match=_NOT_SUPPORTED):
        quantum_relative_entropy(x_var, np.eye(n))
    with pytest.raises(ValueError, match=_NOT_SUPPORTED):
        quantum_relative_entropy(np.eye(n), x_var)


def test_quantum_relative_entropy_rejects_non_affine():
    """Non-constant (including non-affine) CVXPY inputs are rejected."""
    n = 2
    x_var = cvxpy.Variable((n, n), symmetric=True)
    with pytest.raises(ValueError, match=_NOT_SUPPORTED):
        quantum_relative_entropy(x_var @ x_var, cvxpy.Constant(np.eye(n)))


def test_quantum_relative_entropy_numpy_still_works():
    """Numeric numpy path is unaffected by the guard."""
    mat_x = np.diag([0.7, 0.3])
    mat_y = np.diag([0.6, 0.4])
    result = quantum_relative_entropy(mat_x, mat_y)
    assert np.isfinite(result)
    assert result > 0


def test_quantum_relative_entropy_constant_cvxpy_still_works():
    """Constant CVXPY expressions (no free variables) must not be rejected."""
    mat_x = np.diag([0.7, 0.3])
    mat_y = np.diag([0.6, 0.4])
    result = quantum_relative_entropy(cvxpy.Constant(mat_x), cvxpy.Constant(mat_y))
    assert np.isfinite(result)
    assert result > 0
