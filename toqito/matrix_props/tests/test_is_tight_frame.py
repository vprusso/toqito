"""Tests for is_tight_frame."""

import numpy as np
import pytest

from toqito.matrix_props import is_tight_frame


def test_standard_basis_is_tight_frame():
    """Standard basis vectors form a tight frame."""
    e0 = np.array([1, 0])
    e1 = np.array([0, 1])
    assert is_tight_frame([e0, e1])


def test_trine_vectors_are_tight_frame():
    """Mercedes-Benz (trine) vectors form a tight frame in R^2."""
    v0 = np.array([0, 1])
    v1 = np.array([np.sqrt(3) / 2, -1 / 2])
    v2 = np.array([-np.sqrt(3) / 2, -1 / 2])
    assert is_tight_frame([v0, v1, v2])


def test_overcomplete_tight_frame():
    """4 vectors forming a tight frame in R^2."""
    vecs = [
        np.array([1, 0]),
        np.array([0, 1]),
        np.array([1, 0]),
        np.array([0, 1]),
    ]
    assert is_tight_frame(vecs)


def test_non_tight_frame():
    """Two non-orthogonal vectors that don't form a tight frame."""
    v0 = np.array([1, 0])
    v1 = np.array([1, 1]) / np.sqrt(2)
    assert not is_tight_frame([v0, v1])


def test_complex_tight_frame():
    """Complex vectors forming a tight frame."""
    v0 = np.array([1, 0], dtype=complex)
    v1 = np.array([0, 1], dtype=complex)
    v2 = np.array([1, 1j], dtype=complex) / np.sqrt(2)
    v3 = np.array([1, -1j], dtype=complex) / np.sqrt(2)
    assert is_tight_frame([v0, v1, v2, v3])


def test_single_vector_not_tight_frame():
    """A single non-spanning vector is not a tight frame."""
    assert not is_tight_frame([np.array([1, 0, 0])])


def test_empty_list_raises():
    """Empty list should raise ValueError."""
    with pytest.raises(ValueError, match="At least one vector"):
        is_tight_frame([])


def test_inconsistent_dimensions_raises():
    """Vectors of different dimensions should raise ValueError."""
    with pytest.raises(ValueError, match="same dimension"):
        is_tight_frame([np.array([1, 0]), np.array([1, 0, 0])])


def test_zero_vectors_have_zero_frame_bound_and_are_not_tight_frame():
    """A set of zero vectors has frame bound 0 and cannot form a tight frame."""
    assert not is_tight_frame([np.zeros(3), np.zeros(3)])
