"""Tests for tsallis_entropy."""

import re

import cvxpy
import numpy as np
import pytest
from scipy.linalg import logm

from toqito.cones.ln_quantum_entropy import ln_quantum_entropy
from toqito.cones.tsallis_entropy import tsallis_entropy

_NOT_SUPPORTED = re.escape("Affine or variable CVXPY inputs are not yet supported; pass numeric matrices.")


def _tsallis_entropy_reference(mat_x: np.ndarray, t: float) -> float:
    """Compute the Tsallis entropy reference from eigenvalues."""
    if t == 0:
        return float(np.real(-np.trace(mat_x @ logm(mat_x))))
    eigs = np.linalg.eigvalsh((mat_x + mat_x.conj().T) / 2)
    eigs = eigs[eigs > 0]
    return float(np.sum(eigs ** (1 - t) - eigs) / t)


def test_tsallis_entropy_maximally_mixed():
    """Tsallis entropy of the qubit maximally mixed state."""
    rho = np.eye(2) / 2
    t = 0.5
    expected = _tsallis_entropy_reference(rho, t)
    np.testing.assert_allclose(tsallis_entropy(rho, t), expected, rtol=1e-10, atol=1e-10)


def test_tsallis_entropy_t_zero_is_von_neumann():
    """``t = 0`` recovers von Neumann entropy in nats."""
    rng = np.random.default_rng(11)
    g = rng.standard_normal((3, 3))
    mat_x = g @ g.T
    mat_x = mat_x / np.trace(mat_x)
    np.testing.assert_allclose(
        tsallis_entropy(mat_x, 0.0),
        ln_quantum_entropy(mat_x),
        rtol=1e-10,
        atol=1e-10,
    )


def test_tsallis_entropy_constant_cvxpy():
    """Constant CVXPY expressions recurse to the numeric path."""
    mat_x = np.diag([0.25, 0.75])
    t = 0.25
    expr = cvxpy.Constant(mat_x)
    expected = _tsallis_entropy_reference(mat_x, t)
    np.testing.assert_allclose(tsallis_entropy(expr, t), expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("n", [2, 4])
@pytest.mark.parametrize("t", [0.0, 0.25, 0.5, 1.0])
def test_tsallis_entropy_numeric_grid(n: int, t: float) -> None:
    """Random PSD matrices match the eigenvalue reference."""
    seed = n * 100_003 + int(100 * t)
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((n, n))
    mat_x = g @ g.T + 0.1 * np.eye(n)
    mat_x = mat_x / np.trace(mat_x)
    expected = _tsallis_entropy_reference(mat_x, t)
    np.testing.assert_allclose(tsallis_entropy(mat_x, t), expected, rtol=1e-9, atol=1e-9)


def test_tsallis_entropy_rejects_bare_variable():
    """Bare ``Variable`` inputs with ``.value`` set are not supported."""
    mat_x = np.diag([0.25, 0.75])
    x_var = cvxpy.Variable((2, 2), symmetric=True)
    x_var.value = mat_x
    with pytest.raises(ValueError, match=_NOT_SUPPORTED):
        tsallis_entropy(x_var, 0.5)


class TestTsallisEntropyValueErrors:
    """``ValueError`` paths in ``tsallis_entropy``."""

    def test_mat_x_wrong_type(self) -> None:
        """Reject ``mat_x`` that is not a numpy array or CVXPY expression."""
        with pytest.raises(
            ValueError,
            match=re.escape("mat_x must be a numpy array or a cvxpy expression"),
        ):
            tsallis_entropy([[1.0, 0.0], [0.0, 1.0]], 0.5)

    def test_mat_x_not_square(self) -> None:
        """Reject non-square ``mat_x``."""
        with pytest.raises(ValueError, match=re.escape("mat_x must be square.")):
            tsallis_entropy(np.zeros((2, 3)), 0.5)

    def test_t_out_of_range(self) -> None:
        """Reject order parameter ``t`` outside ``[0, 1]``."""
        mat_x = np.eye(2) / 2
        with pytest.raises(ValueError, match=re.escape("t must be in the range [0, 1]")):
            tsallis_entropy(mat_x, 1.5)

    def test_not_positive_semidefinite(self) -> None:
        """Reject non-PSD numeric ``mat_x``."""
        with pytest.raises(ValueError, match=re.escape("mat_x must be a positive semidefinite matrix")):
            tsallis_entropy(np.diag([1.0, -0.1]), 0.5)

    def test_constant_no_value(self) -> None:
        """Reject constant CVXPY expressions with no ``.value``."""
        p = cvxpy.Parameter((2, 2), symmetric=True)
        assert p.value is None
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Constant CVXPY expression has no numeric value; set parameter `.value` "
                "or pass mat_x as a numpy.ndarray."
            ),
        ):
            tsallis_entropy(p, 0.5)

    def test_nonconstant_variable(self) -> None:
        """Reject non-constant CVXPY inputs."""
        x_var = cvxpy.Variable((2, 2), symmetric=True)
        with pytest.raises(ValueError, match=_NOT_SUPPORTED):
            tsallis_entropy(x_var, 0.5)

    def test_nonconstant_quadratic(self) -> None:
        """Reject non-constant quadratic CVXPY inputs."""
        x_var = cvxpy.Variable((2, 2), symmetric=True)
        x_var.value = np.eye(2) / 2
        with pytest.raises(ValueError, match=_NOT_SUPPORTED):
            tsallis_entropy(cvxpy.square(x_var), 0.5)
