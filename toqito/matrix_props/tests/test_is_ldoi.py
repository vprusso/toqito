"""Tests for `is_ldoi`."""

import numpy as np
import pytest

from toqito.matrix_props import is_ldoi


def test_diagonal_state_is_ldoi() -> None:
    """Diagonal bipartite states should satisfy the LDOI property."""
    diag_state = np.diag([1, 2, 3, 4]).astype(np.complex128)
    assert is_ldoi(diag_state)


def test_non_ldoi_returns_false() -> None:
    """A generic non-symmetric matrix should not be LDOI."""
    non_ldoi = np.arange(1, 17, dtype=float).reshape(4, 4)
    assert not is_ldoi(non_ldoi)


def test_is_ldoi_raises_for_nonsquare_matrix() -> None:
    """Input must be a square matrix to evaluate the LDOI property."""
    with pytest.raises(ValueError, match="Input matrix must be square"):
        is_ldoi(np.ones((2, 3)))
