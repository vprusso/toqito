"""Tests for random_linearly_independent_vectors."""

import numpy as np
import pytest

from toqito.matrix_props import is_linearly_independent
from toqito.rand import random_linearly_independent_vectors


@pytest.mark.parametrize(
    "num_vectors, dim, is_real",
    [
        (1, 3, True),
        (2, 3, True),
        (3, 3, True),
        (2, 5, True),
        (1, 3, False),
        (2, 3, False),
        (3, 3, False),
        (4, 4, False),
    ],
)
def test_linearly_independent(num_vectors, dim, is_real):
    """Verify generated vectors are linearly independent."""
    mat = random_linearly_independent_vectors(num_vectors, dim, is_real=is_real, seed=0)
    assert mat.shape == (dim, num_vectors)
    assert np.linalg.matrix_rank(mat) == num_vectors
    assert is_linearly_independent(list(mat.T))


def test_real_vectors_are_real():
    """Real vectors should have no imaginary component."""
    mat = random_linearly_independent_vectors(3, 5, is_real=True, seed=1)
    assert not np.issubdtype(mat.dtype, np.complexfloating)


def test_complex_vectors_are_complex():
    """Complex vectors should have complex dtype."""
    mat = random_linearly_independent_vectors(3, 5, is_real=False, seed=1)
    assert np.issubdtype(mat.dtype, np.complexfloating)


def test_seed_reproducibility():
    """Same seed should produce the same result."""
    a = random_linearly_independent_vectors(3, 5, seed=42)
    b = random_linearly_independent_vectors(3, 5, seed=42)
    np.testing.assert_array_equal(a, b)


def test_different_seeds_differ():
    """Different seeds should produce different results."""
    a = random_linearly_independent_vectors(3, 5, seed=1)
    b = random_linearly_independent_vectors(3, 5, seed=2)
    assert not np.array_equal(a, b)


def test_num_vectors_exceeds_dim_raises():
    """Should raise ValueError when num_vectors > dim."""
    with pytest.raises(ValueError, match="Cannot have more independent vectors"):
        random_linearly_independent_vectors(4, 3)


def test_single_vector():
    """A single vector in any dimension should always be independent."""
    mat = random_linearly_independent_vectors(1, 10, seed=0)
    assert mat.shape == (10, 1)
    assert np.linalg.matrix_rank(mat) == 1


def test_square_case():
    """num_vectors == dim should produce a full-rank square matrix."""
    mat = random_linearly_independent_vectors(5, 5, seed=0)
    assert mat.shape == (5, 5)
    assert np.linalg.matrix_rank(mat) == 5
