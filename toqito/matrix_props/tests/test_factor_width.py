"""Tests for factor_width."""

import numpy as np
import pytest

from toqito.matrix_props import factor_width


@pytest.fixture(scope="module")
def solver_settings():
    return {"eps": 1e-7, "max_iters": 20000}


def test_factor_width_one_for_diagonal(solver_settings):
    mat = np.diag([1.0, 2.0, 0.0])
    result = factor_width(mat, k=1, solver="SCS", solver_kwargs=solver_settings)
    assert result["feasible"] is True
    reconstructed = sum(result["factors"])
    np.testing.assert_allclose(reconstructed, mat, atol=1e-5)


def test_factor_width_false_for_rank_one_sum(solver_settings):
    mat = np.array([[1, 1], [1, 1]], dtype=np.complex128) / 2
    result = factor_width(mat, k=1, solver="SCS", solver_kwargs=solver_settings)
    assert result["feasible"] is False


def test_factor_width_accepts_k_equals_dimension():
    mat = np.eye(3, dtype=np.complex128)
    result = factor_width(mat, k=3)
    assert result["feasible"] is True
    np.testing.assert_allclose(result["factors"][0], mat)


def test_factor_width_zero_matrix_trivial_case():
    mat = np.zeros((2, 2), dtype=np.complex128)
    result = factor_width(mat, k=1)
    assert result["feasible"] is True
    np.testing.assert_allclose(result["factors"][0], np.zeros_like(mat))
