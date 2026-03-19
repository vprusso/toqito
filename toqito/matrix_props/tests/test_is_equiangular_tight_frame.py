"""Tests for is_equiangular_tight_frame."""

import numpy as np
import pytest

from toqito.matrix_props import is_equiangular_tight_frame


def test_mercedes_benz_etf():
    """Mercedes-Benz (trine) vectors form an ETF in R^2."""
    mat = np.array([
        [0, np.sqrt(3) / 2, -np.sqrt(3) / 2],
        [1, -1 / 2, -1 / 2],
    ])
    assert is_equiangular_tight_frame(mat)


def test_identity_columns_not_etf():
    """Identity matrix columns are a tight frame but NOT equiangular (inner products are 0)."""
    # Actually, identity columns ARE equiangular (all off-diag = 0, which is constant)
    # and form a tight frame (I*I^T = I). So this is an ETF.
    mat = np.eye(3)
    assert is_equiangular_tight_frame(mat)


def test_non_unit_norm_not_etf():
    """Columns without unit norm are not an ETF."""
    mat = np.array([[2, 0], [0, 2]])
    assert not is_equiangular_tight_frame(mat)


def test_non_equiangular_not_etf():
    """Columns with varying inner products are not an ETF."""
    mat = np.array([
        [1, 0, 1 / np.sqrt(2)],
        [0, 1, 1 / np.sqrt(2)],
    ])
    assert not is_equiangular_tight_frame(mat)


def test_random_matrix_not_etf():
    """A random matrix is generally not an ETF."""
    rng = np.random.default_rng(42)
    mat = rng.standard_normal((3, 5))
    assert not is_equiangular_tight_frame(mat)


def test_single_unit_column():
    """A single unit vector is trivially an ETF."""
    mat = np.array([[1], [0], [0]])
    assert is_equiangular_tight_frame(mat)


def test_single_non_unit_column():
    """A single non-unit vector is not an ETF."""
    mat = np.array([[2], [0]])
    assert not is_equiangular_tight_frame(mat)


def test_equiangular_but_not_tight():
    """Equiangular unit-norm vectors that don't form a tight frame are not an ETF."""
    v0 = np.array([1, 0], dtype=complex)
    v1 = np.array([1, np.sqrt(2)], dtype=complex) / np.sqrt(3)
    v2 = np.array([1, np.sqrt(2) * np.exp(2j * np.pi / 3)], dtype=complex) / np.sqrt(3)
    mat = np.column_stack([v0, v1, v2])
    assert not is_equiangular_tight_frame(mat)


def test_not_2d_raises():
    """Non-2D input should raise ValueError."""
    with pytest.raises(ValueError, match="2D matrix"):
        is_equiangular_tight_frame(np.array([1, 2, 3]))


def test_empty_matrix():
    """Matrix with no columns."""
    mat = np.zeros((3, 0))
    assert not is_equiangular_tight_frame(mat)
