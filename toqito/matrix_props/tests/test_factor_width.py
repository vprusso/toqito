"""Tests for factor_width."""

import numpy as np
import pytest

from toqito.matrix_props import factor_width

pytestmark = [
    pytest.mark.filterwarnings("ignore:Converting A to a CSC"),
    pytest.mark.filterwarnings("ignore:Converting P to a CSC"),
    pytest.mark.filterwarnings("ignore:Initializing a Constant with a nested list"),
    pytest.mark.filterwarnings("ignore:Solution may be inaccurate"),
]


@pytest.fixture(scope="module")
def solver_settings():
    """Provide tighter solver tolerances for deterministic tests."""
    return {"eps": 1e-7, "max_iters": 20000}


def test_factor_width_one_for_diagonal(solver_settings):
    """Diagonal matrix with zeros is 1-factorable."""
    mat = np.diag([1.0, 2.0, 0.0])
    result = factor_width(mat, k=1, solver="SCS", solver_kwargs=solver_settings)
    assert result["feasible"] is True
    reconstructed = sum(result["factors"])
    np.testing.assert_allclose(reconstructed, mat, atol=1e-5)


def test_factor_width_false_for_rank_one_sum(solver_settings):
    """Rank-one projector should not be 1-factorable."""
    mat = np.array([[1, 1], [1, 1]], dtype=np.complex128) / 2
    result = factor_width(mat, k=1, solver="SCS", solver_kwargs=solver_settings)
    assert result["feasible"] is False


def test_factor_width_accepts_k_equals_dimension():
    """Factor width equals dimension yields trivial decomposition."""
    mat = np.eye(3, dtype=np.complex128)
    result = factor_width(mat, k=3)
    assert result["feasible"] is True
    np.testing.assert_allclose(result["factors"][0], mat)


def test_factor_width_zero_matrix_trivial_case():
    """Zero matrix should always be feasible."""
    mat = np.zeros((2, 2), dtype=np.complex128)
    result = factor_width(mat, k=1)
    assert result["feasible"] is True
    np.testing.assert_allclose(result["factors"][0], np.zeros_like(mat))


def test_factor_width_ball_example(solver_settings):
    """Verify factor width on the ensemble described in the manuscript."""
    d = 4
    k = 2
    x = 0.4
    eps = x / (4 * d)
    u = np.zeros((d, 1), dtype=np.complex128)
    u[:2, 0] = 1 / np.sqrt(2)
    uu = u @ u.conj().T
    mat = (1 - eps / d) * (x * np.eye(d) / d + (1 - x) * uu)
    result = factor_width(mat, k=k, solver="SCS", solver_kwargs=solver_settings)
    assert result["feasible"] is True
    reconstructed = sum(result["factors"])
    np.testing.assert_allclose(reconstructed, mat, atol=1e-5)
