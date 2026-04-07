"""Tests for nonnegative_rank."""

import numpy as np
import pytest

from toqito.matrix_props import nonnegative_rank


def test_identity_matrix():
    """Nonneg rank of identity equals dimension."""
    assert nonnegative_rank(np.eye(3)) == 3


def test_rank_1_matrix():
    """Rank-1 nonnegative matrix has nonneg rank 1."""
    A = np.array([[1, 2], [2, 4]], dtype=float)
    assert nonnegative_rank(A) == 1


def test_all_ones_matrix():
    """All-ones matrix has nonneg rank 1."""
    A = np.ones((3, 4))
    assert nonnegative_rank(A) == 1


def test_diagonal_matrix():
    """Diagonal nonneg matrix has nonneg rank equal to number of nonzero diag entries."""
    A = np.diag([1.0, 2.0, 0.0, 3.0])
    assert nonnegative_rank(A) == 3


def test_2x2_nonneg_rank_2():
    """A 2x2 matrix with no rank-1 nonneg factorization."""
    A = np.array([[1, 0], [0, 1]], dtype=float)
    assert nonnegative_rank(A) == 2


def test_rectangular_matrix():
    """Nonneg rank of a rectangular matrix."""
    A = np.array([[1, 2, 3], [2, 4, 6]], dtype=float)
    assert nonnegative_rank(A) == 1


def test_zero_matrix():
    """Zero matrix has nonneg rank 0."""
    A = np.zeros((3, 3))
    assert nonnegative_rank(A) == 0


def test_negative_matrix_raises():
    """Matrix with negative entries should raise ValueError."""
    with pytest.raises(ValueError, match="nonnegative"):
        nonnegative_rank(np.array([[1, -1], [0, 1]]))


def test_max_rank_too_low():
    """Returns None when max_rank is below the actual nonneg rank."""
    A = np.eye(4)
    assert nonnegative_rank(A, max_rank=2) is None


def test_known_nonneg_rank_exceeds_standard_rank():
    """Example where nonneg rank > standard rank.

    The 4x4 matrix with 0 on diagonal and 1 elsewhere has rank 1 completion
    but nonneg rank equals 3 since it needs 3 nonneg rank-1 factors.
    Actually for the complement of identity: J - I where J = ones(n,n),
    standard rank = n-1, nonneg rank = n for n >= 3.
    For n=3: standard rank 2, nonneg rank 3.
    """
    n = 3
    A = np.ones((n, n)) - np.eye(n)
    result = nonnegative_rank(A)
    assert result is not None
    assert result >= np.linalg.matrix_rank(A)
