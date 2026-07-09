"""Tests for integral_relative_entropy."""

import re

import cvxpy
import numpy as np
import pytest
from scipy.linalg import LinAlgError

from toqito.channels import depolarizing
from toqito.cones.integral_relative_entropy import (
    _generalized_eigenvalues,
    _make_delta,
    _make_gamma,
    _make_grid,
    _sandwich_parameters,
    evaluate_relative_entropy_integral,
)


def test_generalized_eigenvalues_eig_fallback():
    """``eigh`` may fail when the right factor is singular; ``eig`` is used instead."""
    choi_1 = np.asarray(depolarizing(2, 1))
    choi_2 = np.asarray(depolarizing(2, 0.2))
    w_yx = _generalized_eigenvalues(choi_2, choi_1)
    assert np.any(np.isfinite(w_yx))


def test_sandwich_parameters_no_finite_eigenvalues(monkeypatch):
    """All non-finite pencil eigenvalues should raise ``ValueError``."""

    def all_infinite(*args, **kwargs):
        return np.array([np.inf, np.inf])

    monkeypatch.setattr(
        "toqito.cones.integral_relative_entropy._generalized_eigenvalues",
        all_infinite,
    )

    with pytest.raises(
        ValueError,
        match=re.escape("Failed to compute sandwich parameters from generalized eigenvalues."),
    ):
        _sandwich_parameters(np.eye(2), np.eye(2) / 2)


def test_make_grid_endpoints():
    """Grid should start at ``mu`` and end at ``lam``."""
    mu, lam, epsilon = 0.5, 2.0, 0.01
    grid = _make_grid(mu, lam, epsilon)
    assert grid[0] == pytest.approx(mu)
    assert grid[-1] == pytest.approx(lam)
    assert len(grid) >= 2


def test_make_gamma_and_delta_match_grid_length():
    """Coefficient vectors should match the grid length."""
    grid = _make_grid(0.25, 1.5, 1e-2)
    gamma = _make_gamma(grid)
    delta = _make_delta(grid)
    assert len(gamma) == len(grid)
    assert len(delta) == len(grid)


def test_evaluate_relative_entropy_integral_shape_mismatch():
    """Mismatched shapes should raise ``ValueError``."""
    with pytest.raises(ValueError, match="must have the same shape"):
        evaluate_relative_entropy_integral(np.eye(2), np.eye(3))


def test_evaluate_relative_entropy_integral_mat_x_not_psd():
    """Non-PSD ``mat_x`` should raise ``ValueError``."""
    mat_x = np.array([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(ValueError, match="mat_x must be a positive semidefinite matrix"):
        evaluate_relative_entropy_integral(mat_x, np.eye(2))


def test_evaluate_relative_entropy_integral_mat_y_not_psd():
    """Non-PSD ``mat_y`` should raise ``ValueError``."""
    mat_y = np.array([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(ValueError, match="mat_y must be a positive semidefinite matrix"):
        evaluate_relative_entropy_integral(np.eye(2), mat_y)


def test_evaluate_relative_entropy_integral_identical_inputs():
    """Identical inputs should return zero without solving SDPs."""
    mat = np.eye(2) / 2
    assert evaluate_relative_entropy_integral(mat, mat) == 0.0
    assert evaluate_relative_entropy_integral(mat, mat, mean=False) == (0.0, 0.0)


def test_evaluate_relative_entropy_integral_degenerate_sandwich(monkeypatch):
    """Degenerate ``mu``/``lambda`` should raise ``ValueError``."""
    monkeypatch.setattr(
        "toqito.cones.integral_relative_entropy._sandwich_parameters",
        lambda mat_x, mat_y: (1.0, 1.0),
    )
    with pytest.raises(ValueError, match="0 < mu < lambda"):
        evaluate_relative_entropy_integral(np.eye(2) / 2, np.eye(2) / 4)


def test_evaluate_relative_entropy_integral_mean_false():
    """``mean=False`` should return separate lower and upper bounds."""
    mat_x = np.diag([0.7, 0.3])
    mat_y = np.diag([0.4, 0.6])
    lower, upper = evaluate_relative_entropy_integral(mat_x, mat_y, mean=False)
    assert np.isfinite(lower)
    assert np.isfinite(upper)
    assert upper >= lower - 1e-8


def test_evaluate_relative_entropy_integral_complex_hermitian():
    """Complex Hermitian inputs should use Hermitian CVXPY variables."""
    mat_x = np.diag([0.6, 0.4]).astype(complex)
    mat_y = np.diag([0.5, 0.5]).astype(complex)
    result = evaluate_relative_entropy_integral(mat_x, mat_y)
    assert np.isfinite(result)
    assert result >= 0.0


def test_evaluate_relative_entropy_integral_lower_sdp_failure(monkeypatch):
    """A failed lower-bound solve should raise ``RuntimeError``."""

    class FakeProblem:
        created = 0

        def __init__(self, objective, constraints):
            self.value = 1.0
            FakeProblem.created += 1
            self.status = cvxpy.INFEASIBLE if FakeProblem.created == 1 else cvxpy.OPTIMAL

        def solve(self, **kwargs):
            pass

    FakeProblem.created = 0
    monkeypatch.setattr(cvxpy, "Problem", FakeProblem)

    with pytest.raises(RuntimeError, match="Lower-bound SDP failed"):
        evaluate_relative_entropy_integral(np.diag([0.7, 0.3]), np.diag([0.4, 0.6]))


def test_evaluate_relative_entropy_integral_upper_sdp_failure(monkeypatch):
    """A failed upper-bound solve should raise ``RuntimeError``."""

    class FakeProblem:
        created = 0

        def __init__(self, objective, constraints):
            self.value = 1.0
            FakeProblem.created += 1
            if FakeProblem.created == 1:
                self.status = cvxpy.OPTIMAL
            else:
                self.status = cvxpy.INFEASIBLE

        def solve(self, **kwargs):
            pass

    FakeProblem.created = 0
    monkeypatch.setattr(cvxpy, "Problem", FakeProblem)

    with pytest.raises(RuntimeError, match="Upper-bound SDP failed"):
        evaluate_relative_entropy_integral(np.diag([0.7, 0.3]), np.diag([0.4, 0.6]))


def test_evaluate_relative_entropy_integral_warns_on_inaccurate_lower(monkeypatch):
    """``OPTIMAL_INACCURATE`` on the lower SDP should warn."""

    class FakeProblem:
        created = 0

        def __init__(self, objective, constraints):
            self.value = 0.1
            FakeProblem.created += 1
            self.status = cvxpy.OPTIMAL_INACCURATE if FakeProblem.created == 1 else cvxpy.OPTIMAL

        def solve(self, **kwargs):
            pass

    FakeProblem.created = 0
    monkeypatch.setattr(cvxpy, "Problem", FakeProblem)

    with pytest.warns(UserWarning, match="Lower-bound SDP returned OPTIMAL_INACCURATE"):
        evaluate_relative_entropy_integral(np.diag([0.7, 0.3]), np.diag([0.4, 0.6]))


def test_evaluate_relative_entropy_integral_warns_on_inaccurate_upper(monkeypatch):
    """``OPTIMAL_INACCURATE`` on the upper SDP should warn."""

    class FakeProblem:
        created = 0

        def __init__(self, objective, constraints):
            self.value = 0.1
            FakeProblem.created += 1
            self.status = cvxpy.OPTIMAL if FakeProblem.created == 1 else cvxpy.OPTIMAL_INACCURATE

        def solve(self, **kwargs):
            pass

    FakeProblem.created = 0
    monkeypatch.setattr(cvxpy, "Problem", FakeProblem)

    with pytest.warns(UserWarning, match="Upper-bound SDP returned OPTIMAL_INACCURATE"):
        evaluate_relative_entropy_integral(np.diag([0.7, 0.3]), np.diag([0.4, 0.6]))


def test_sandwich_parameters_raises_on_lin_alg_error(monkeypatch):
    """``LinAlgError`` from the pencil solver should be wrapped in ``ValueError``."""

    def failing(*args, **kwargs):
        raise LinAlgError("singular pencil")

    monkeypatch.setattr(
        "toqito.cones.integral_relative_entropy._generalized_eigenvalues",
        failing,
    )

    with pytest.raises(
        ValueError,
        match=re.escape("Failed to compute sandwich parameters from generalized eigenvalues."),
    ):
        _sandwich_parameters(np.eye(2), np.eye(2) / 2)


def test_evaluate_relative_entropy_integral_numpy_still_works():
    """Numeric numpy path is unaffected by the guard."""
    mat_x = np.diag([0.7, 0.3])
    mat_y = np.diag([0.6, 0.4])
    result = evaluate_relative_entropy_integral(mat_x, mat_y)
    assert np.isfinite(result)


def test_evaluate_relative_entropy_integral_free_variable_mat_x_raises():
    """A free CVXPY Variable in mat_x must raise ValueError mentioning free CVXPY variables."""
    x_var = cvxpy.Variable((2, 2), symmetric=True)
    x_var.value = np.diag([0.7, 0.3])
    mat_y = np.diag([0.6, 0.4])
    with pytest.raises(ValueError, match="free CVXPY variables"):
        evaluate_relative_entropy_integral(x_var, mat_y)


def test_evaluate_relative_entropy_integral_free_variable_mat_y_raises():
    """A free CVXPY Variable in mat_y must raise ValueError mentioning free CVXPY variables."""
    mat_x = np.diag([0.7, 0.3])
    y_var = cvxpy.Variable((2, 2), symmetric=True)
    y_var.value = np.diag([0.6, 0.4])
    with pytest.raises(ValueError, match="free CVXPY variables"):
        evaluate_relative_entropy_integral(mat_x, y_var)
