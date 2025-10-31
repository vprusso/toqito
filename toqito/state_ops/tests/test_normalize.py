"""Tests for the normalize state operation."""

import numpy as np
import pytest

from toqito.state_ops import normalize


def test_normalize_returns_unit_vector():
    """Normalization produces a unit vector."""
    state = np.array([2, 0], dtype=np.complex128)
    result = normalize(state)
    assert np.isclose(np.linalg.norm(result), 1.0)


def test_normalize_flattens_column_vector():
    """Column vectors are flattened during normalization."""
    column = np.array([[1.0], [1.0]], dtype=np.complex128)
    result = normalize(column)
    expected = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2)
    np.testing.assert_allclose(result, expected)


def test_normalize_rejects_matrix_shape():
    """Matrix-shaped inputs raise a ValueError."""
    with pytest.raises(ValueError):
        normalize(np.eye(2, dtype=np.complex128))


def test_normalize_rejects_zero_vector():
    """Zero vectors cannot be normalized."""
    with pytest.raises(ValueError):
        normalize(np.zeros(3, dtype=np.complex128))
