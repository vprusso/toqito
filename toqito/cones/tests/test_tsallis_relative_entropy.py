"""Tests for tsallis_relative_entropy."""

import re

import cvxpy
import numpy as np
import pytest

from toqito.cones.lieb_ando import lieb_ando
from toqito.cones.tsallis_relative_entropy import tsallis_relative_entropy
from toqito.state_props.quantum_relative_entropy import quantum_relative_entropy

_NOT_SUPPORTED = re.escape(
    "Affine or variable CVXPY inputs are not yet supported; pass numeric matrices."
)


def _tsallis_relative_entropy_reference(
    mat_x: np.ndarray,
    mat_y: np.ndarray,
    t: float,
) -> float:
    """Compute the reference value using ``lieb_ando`` for the cross term."""
    if t == 0:
        return quantum_relative_entropy(mat_x, mat_y)
    n = int(mat_x.shape[0])
    mat_x = (mat_x + mat_x.conj().T) / 2
    mat_y = (mat_y + mat_y.conj().T) / 2
    trace_cross = lieb_ando(mat_x, mat_y, np.eye(n), t)
    return float(np.real((np.trace(mat_x) - trace_cross) / t))


def test_tsallis_relative_entropy_equal_states():
    """Identical states give zero Tsallis relative entropy."""
    rho = np.eye(2) / 2
    np.testing.assert_allclose(
        tsallis_relative_entropy(rho, rho, 0.5),
        0.0,
        atol=1e-10,
    )


def test_tsallis_relative_entropy_t_zero_is_quantum_relative_entropy():
    """``t = 0`` recovers quantum relative entropy in nats."""
    rng = np.random.default_rng(13)
    g = rng.standard_normal((3, 3))
    mat_x = g @ g.T
    mat_x = mat_x / np.trace(mat_x)
    h = rng.standard_normal((3, 3))
    mat_y = h @ h.T + 0.2 * np.eye(3)
    mat_y = mat_y / np.trace(mat_y)
    np.testing.assert_allclose(
        tsallis_relative_entropy(mat_x, mat_y, 0.0),
        quantum_relative_entropy(mat_x, mat_y),
        rtol=1e-10,
        atol=1e-10,
    )


def test_tsallis_relative_entropy_numpy_with_constant_cvxpy():
    """Mixed numpy and constant CVXPY inputs promote ``mat_x`` to ``Constant``."""
    mat_x = np.diag([0.25, 0.75])
    mat_y = np.diag([0.5, 0.5])
    t = 0.25
    expected = _tsallis_relative_entropy_reference(mat_x, mat_y, t)
    got = tsallis_relative_entropy(mat_x, cvxpy.Constant(mat_y), t)
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-10)


def test_tsallis_relative_entropy_constant_cvxpy_with_numpy():
    """Constant CVXPY expressions recurse to the numeric path."""
    mat_x = np.diag([0.25, 0.75])
    mat_y = np.diag([0.5, 0.5])
    t = 0.25
    expected = _tsallis_relative_entropy_reference(mat_x, mat_y, t)
    got = tsallis_relative_entropy(cvxpy.Constant(mat_x), cvxpy.Constant(mat_y), t)
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("n", [2, 4])
@pytest.mark.parametrize("t", [0.0, 0.25, 0.5, 1.0])
def test_tsallis_relative_entropy_numeric_grid(n: int, t: float) -> None:
    """Random PSD pairs match the Lieb--Ando reference."""
    seed = n * 100_003 + int(100 * t)
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((n, n))
    mat_x = g @ g.T + 0.1 * np.eye(n)
    mat_x = mat_x / np.trace(mat_x)
    h = rng.standard_normal((n, n))
    mat_y = h @ h.T + 0.1 * np.eye(n)
    mat_y = mat_y / np.trace(mat_y)
    expected = _tsallis_relative_entropy_reference(mat_x, mat_y, t)
    np.testing.assert_allclose(tsallis_relative_entropy(mat_x, mat_y, t), expected, rtol=1e-9, atol=1e-9)


def test_tsallis_relative_entropy_rejects_bare_variable():
    """Bare ``Variable`` inputs with ``.value`` set are not supported (reviewer case)."""
    mat_a = np.diag([0.7, 0.3])
    mat_b = np.diag([0.6, 0.4])
    x_var = cvxpy.Variable((2, 2), symmetric=True)
    x_var.value = mat_a
    with pytest.raises(ValueError, match=_NOT_SUPPORTED):
        tsallis_relative_entropy(x_var, mat_b, 0.5)


def test_tsallis_relative_entropy_rejects_nonconstant_affine():
    """Non-constant affine expressions are rejected."""
    mat_x = np.diag([0.25, 0.75])
    w_var = cvxpy.Variable((2, 2), symmetric=True)
    w_var.value = np.zeros((2, 2))
    expr = cvxpy.Constant(mat_x) + w_var - w_var
    with pytest.raises(ValueError, match=_NOT_SUPPORTED):
        tsallis_relative_entropy(expr, np.eye(2) / 2, 0.5)


def test_tsallis_relative_entropy_commuting_reference():
    """Commuting diagonal states match the eigenvalue formula."""
    mat_x = np.diag([0.2, 0.8])
    mat_y = np.diag([0.4, 0.6])
    t = 0.5
    eigs_x = np.diag(mat_x)
    eigs_y = np.diag(mat_y)
    expected = float(np.sum(eigs_x - eigs_x ** (1 - t) * eigs_y**t) / t)
    np.testing.assert_allclose(tsallis_relative_entropy(mat_x, mat_y, t), expected, rtol=1e-10)


class TestTsallisRelativeEntropyValueErrors:
    """``ValueError`` paths in ``tsallis_relative_entropy``."""

    def test_mat_x_wrong_type(self) -> None:
        """Reject ``mat_x`` that is not a numpy array or CVXPY expression."""
        with pytest.raises(
            ValueError,
            match=re.escape("mat_x must be a numpy array or a cvxpy expression"),
        ):
            tsallis_relative_entropy([[1.0, 0.0], [0.0, 1.0]], np.eye(2), 0.5)

    def test_mat_y_wrong_type(self) -> None:
        """Reject ``mat_y`` that is not a numpy array or CVXPY expression."""
        with pytest.raises(
            ValueError,
            match=re.escape("mat_y must be a numpy array or a cvxpy expression"),
        ):
            tsallis_relative_entropy(np.eye(2), [[1.0, 0.0], [0.0, 1.0]], 0.5)

    def test_shape_mismatch(self) -> None:
        """Reject ``mat_x`` and ``mat_y`` with different shapes."""
        with pytest.raises(ValueError, match=re.escape("mat_x and mat_y must have the same shape")):
            tsallis_relative_entropy(np.eye(2), np.eye(3), 0.5)

    def test_t_out_of_range(self) -> None:
        """Reject order parameter ``t`` outside ``[0, 1]``."""
        with pytest.raises(ValueError, match=re.escape("t must be in the range [0, 1]")):
            tsallis_relative_entropy(np.eye(2) / 2, np.eye(2) / 2, 1.5)

    def test_not_positive_semidefinite(self) -> None:
        """Reject non-PSD numeric ``mat_x``."""
        with pytest.raises(ValueError, match=re.escape("mat_x must be a positive semidefinite matrix")):
            tsallis_relative_entropy(np.diag([1.0, -0.1]), np.eye(2) / 2, 0.5)

    def test_mat_y_not_positive_semidefinite(self) -> None:
        """Reject non-PSD numeric ``mat_y``."""
        with pytest.raises(ValueError, match=re.escape("mat_y must be a positive semidefinite matrix")):
            tsallis_relative_entropy(np.eye(2) / 2, np.diag([1.0, -0.1]), 0.5)

    def test_constant_no_value(self) -> None:
        """Reject constant CVXPY expressions with no ``.value``."""
        p = cvxpy.Parameter((2, 2), symmetric=True)
        assert p.value is None
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Constant CVXPY expression has no numeric value; set parameter `.value` or pass a numpy.ndarray."
            ),
        ):
            tsallis_relative_entropy(p, cvxpy.Constant(np.eye(2) / 2), 0.5)

    def test_nonconstant_variable(self) -> None:
        """Reject non-constant CVXPY inputs."""
        x_var = cvxpy.Variable((2, 2), symmetric=True)
        x_var.value = np.eye(2) / 2
        y_c = cvxpy.Constant(np.eye(2) / 2)
        with pytest.raises(ValueError, match=_NOT_SUPPORTED):
            tsallis_relative_entropy(x_var, y_c, 0.5)

    def test_nonconstant_quadratic(self) -> None:
        """Reject non-constant quadratic CVXPY inputs."""
        x_var = cvxpy.Variable((2, 2), symmetric=True)
        x_var.value = np.eye(2) / 2
        y_c = cvxpy.Constant(np.eye(2) / 2)
        with pytest.raises(ValueError, match=_NOT_SUPPORTED):
            tsallis_relative_entropy(cvxpy.square(x_var), y_c, 0.5)
