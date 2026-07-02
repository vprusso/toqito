"""Tests for tsallis_relative_entropy."""

import re

import cvxpy
import numpy as np
import pytest

from toqito.cones.lieb_ando import lieb_ando
from toqito.cones.tsallis_relative_entropy import tsallis_relative_entropy
from toqito.matrix_props import is_positive_semidefinite
from toqito.state_props.quantum_relative_entropy import quantum_relative_entropy


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


def test_tsallis_relative_entropy_constant_cvxpy():
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
    np.testing.assert_allclose(
        tsallis_relative_entropy(mat_x, mat_y, t), expected, rtol=1e-9, atol=1e-9
    )


def _affine_fixed_at(mat: np.ndarray) -> cvxpy.Expression:
    """``Constant(A) + W - W`` with ``W.value = 0`` (algebraically ``A``)."""
    n = mat.shape[0]
    if np.any(np.imag(mat) != 0):
        w_var = cvxpy.Variable((n, n), hermitian=True)
        w_var.value = np.zeros((n, n), dtype=np.complex128)
    else:
        w_var = cvxpy.Variable((n, n), symmetric=True)
        w_var.value = np.zeros((n, n))
    return cvxpy.Constant(mat) + w_var - w_var


@pytest.mark.parametrize("n", [2, 4])
@pytest.mark.parametrize("t", [0.0, 0.25, 0.5, 1.0])
@pytest.mark.parametrize(
    ("x_affine", "y_affine"),
    [
        (True, True),
        (True, False),
        (False, True),
    ],
    ids=["both_affine", "x_affine_y_constant", "x_constant_y_affine"],
)
def test_tsallis_relative_entropy_affine_grid(
    n: int,
    t: float,
    x_affine: bool,
    y_affine: bool,
) -> None:
    """Affine SDP branch matches the numeric reference at fixed matrix values."""
    seed = n * 100_003 + int(100 * t) + int(x_affine) * 2 + int(y_affine)
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((n, n))
    mat_x = g @ g.T + 0.1 * np.eye(n)
    mat_x = mat_x / np.trace(mat_x)
    h = rng.standard_normal((n, n))
    mat_y = h @ h.T + 0.1 * np.eye(n)
    mat_y = mat_y / np.trace(mat_y)
    expected = _tsallis_relative_entropy_reference(mat_x, mat_y, t)

    x_expr = _affine_fixed_at(mat_x) if x_affine else cvxpy.Constant(mat_x)
    y_expr = _affine_fixed_at(mat_y) if y_affine else cvxpy.Constant(mat_y)

    val = tsallis_relative_entropy(x_expr, y_expr, t)
    if t == 0:
        np.testing.assert_allclose(val, expected, rtol=1e-4, atol=1e-4)
    else:
        assert val == pytest.approx(expected, rel=0, abs=1e-2)


def test_tsallis_relative_entropy_numpy_with_affine():
    """A numpy ``mat_x`` with affine ``mat_y`` promotes ``mat_x`` to ``Constant``."""
    mat_x = np.diag([0.25, 0.75])
    mat_y = np.diag([0.4, 0.6])
    mat_y = mat_y / np.trace(mat_y)
    t = 0.5
    expected = _tsallis_relative_entropy_reference(mat_x, mat_y, t)
    val = tsallis_relative_entropy(mat_x, _affine_fixed_at(mat_y), t)
    assert val == pytest.approx(expected, rel=0, abs=1e-2)


def test_tsallis_relative_entropy_affine_with_numpy():
    """An affine ``mat_x`` with numpy ``mat_y`` promotes ``mat_y`` to ``Constant``."""
    mat_x = np.diag([0.25, 0.75])
    mat_y = np.diag([0.4, 0.6])
    mat_y = mat_y / np.trace(mat_y)
    t = 0.5
    expected = _tsallis_relative_entropy_reference(mat_x, mat_y, t)
    val = tsallis_relative_entropy(_affine_fixed_at(mat_x), mat_y, t)
    assert val == pytest.approx(expected, rel=0, abs=1e-2)


def test_tsallis_relative_entropy_affine_hermitian_sdp():
    """Hermitian affine inputs exercise the complex Hermitian-variable SDP path."""
    n = 2
    t = 0.5
    rng = np.random.default_rng(19)
    g = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    h0 = g @ g.conj().T + 0.5 * np.eye(n)
    h0 = (h0 + h0.conj().T) / 2
    mat_x = h0 / np.trace(h0)
    k = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    h1 = k @ k.conj().T + 0.5 * np.eye(n)
    h1 = (h1 + h1.conj().T) / 2
    mat_y = h1 / np.trace(h1)
    assert is_positive_semidefinite(np.asarray(mat_x, dtype=np.complex128))
    assert is_positive_semidefinite(np.asarray(mat_y, dtype=np.complex128))
    w_x = cvxpy.Variable((n, n), hermitian=True)
    w_x.value = np.zeros((n, n), dtype=np.complex128)
    w_y = cvxpy.Variable((n, n), hermitian=True)
    w_y.value = np.zeros((n, n), dtype=np.complex128)
    x_expr = cvxpy.Constant(mat_x) + w_x - w_x
    y_expr = cvxpy.Constant(mat_y) + w_y - w_y
    expected = _tsallis_relative_entropy_reference(mat_x, mat_y, t)
    val = tsallis_relative_entropy(x_expr, y_expr, t)
    assert val == pytest.approx(expected, rel=0, abs=1e-2)


def test_tsallis_relative_entropy_sdp_failure(monkeypatch):
    """A failed SDP solve should raise ``ValueError``."""

    class FakeProblem:
        def __init__(self, objective, constraints):
            self.value = 1.0
            self.status = cvxpy.INFEASIBLE

        def solve(self, **kwargs):
            pass

    monkeypatch.setattr(cvxpy, "Problem", FakeProblem)

    n = 2
    mat_x = np.eye(n) / n
    mat_y = np.diag([0.4, 0.6])
    w_var = cvxpy.Variable((n, n), symmetric=True)
    w_var.value = np.zeros((n, n))
    with pytest.raises(
        ValueError, match=re.escape("The SDP did not solve successfully")
    ):
        tsallis_relative_entropy(cvxpy.Constant(mat_x) + w_var - w_var, mat_y, 0.5)


def test_tsallis_relative_entropy_commuting_reference():
    """Commuting diagonal states match the eigenvalue formula."""
    mat_x = np.diag([0.2, 0.8])
    mat_y = np.diag([0.4, 0.6])
    t = 0.5
    eigs_x = np.diag(mat_x)
    eigs_y = np.diag(mat_y)
    expected = float(np.sum(eigs_x - eigs_x ** (1 - t) * eigs_y**t) / t)
    np.testing.assert_allclose(
        tsallis_relative_entropy(mat_x, mat_y, t), expected, rtol=1e-10
    )


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
        with pytest.raises(
            ValueError, match=re.escape("mat_x and mat_y must have the same shape")
        ):
            tsallis_relative_entropy(np.eye(2), np.eye(3), 0.5)

    def test_t_out_of_range(self) -> None:
        """Reject order parameter ``t`` outside ``[0, 1]``."""
        with pytest.raises(
            ValueError, match=re.escape("t must be in the range [0, 1]")
        ):
            tsallis_relative_entropy(np.eye(2) / 2, np.eye(2) / 2, 1.5)

    def test_not_positive_semidefinite(self) -> None:
        """Reject non-PSD numeric ``mat_x``."""
        with pytest.raises(
            ValueError, match=re.escape("mat_x must be a positive semidefinite matrix")
        ):
            tsallis_relative_entropy(np.diag([1.0, -0.1]), np.eye(2) / 2, 0.5)

    def test_mat_y_not_positive_semidefinite(self) -> None:
        """Reject non-PSD numeric ``mat_y``."""
        with pytest.raises(
            ValueError, match=re.escape("mat_y must be a positive semidefinite matrix")
        ):
            tsallis_relative_entropy(np.eye(2) / 2, np.diag([1.0, -0.1]), 0.5)

    def test_constant_no_value(self) -> None:
        """Reject constant CVXPY expressions with no ``.value``."""
        p = cvxpy.Parameter((2, 2), symmetric=True)
        assert p.value is None
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Constant CVXPY expression has no numeric value; set parameter `.value` "
                "or pass a numpy.ndarray."
            ),
        ):
            tsallis_relative_entropy(p, cvxpy.Constant(np.eye(2) / 2), 0.5)

    def test_not_affine(self) -> None:
        """Reject non-affine CVXPY inputs."""
        x_var = cvxpy.Variable((2, 2), symmetric=True)
        x_var.value = np.eye(2) / 2
        y_c = cvxpy.Constant(np.eye(2) / 2)
        with pytest.raises(
            ValueError,
            match=re.escape("mat_x and mat_y must be affine CVXPY expressions."),
        ):
            tsallis_relative_entropy(cvxpy.square(x_var), y_c, 0.5)

    def test_affine_missing_initial_value(self) -> None:
        """Reject affine inputs with no initial ``.value`` for PSD checks."""
        x_var = cvxpy.Variable((2, 2), symmetric=True)
        y_c = cvxpy.Constant(np.eye(2) / 2)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Affine mat_x and mat_y need numeric initial values; set `.value` for PSD checks."
            ),
        ):
            tsallis_relative_entropy(x_var, y_c, 0.5)

    def test_affine_not_psd_at_initial_value(self) -> None:
        """Reject affine ``mat_x`` that is not PSD at ``.value``."""
        x_var = cvxpy.Variable((2, 2), symmetric=True)
        x_var.value = np.diag([1.0, -0.1])
        y_c = cvxpy.Constant(np.eye(2) / 2)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "mat_x must be positive semidefinite at the initial value."
            ),
        ):
            tsallis_relative_entropy(x_var, y_c, 0.5)

    def test_affine_mat_y_not_psd_at_initial_value(self) -> None:
        """Reject affine ``mat_y`` that is not PSD at ``.value``."""
        x_c = cvxpy.Constant(np.eye(2) / 2)
        y_var = cvxpy.Variable((2, 2), symmetric=True)
        y_var.value = np.diag([1.0, -0.1])
        with pytest.raises(
            ValueError,
            match=re.escape(
                "mat_y must be positive semidefinite at the initial value."
            ),
        ):
            tsallis_relative_entropy(x_c, y_var, 0.5)
