"""Tests for entangled_subspace."""

import numpy as np
import pytest

from toqito.matrices import entangled_subspace
from toqito.state_props import schmidt_rank


@pytest.mark.parametrize(
    "dim, local_dim, r",
    [
        (1, 3, 1),
        (2, 3, 1),
        (4, 3, 1),
        (1, 4, 1),
        (4, 4, 1),
        (9, 4, 1),
        (1, [3, 4], 1),
        (4, [3, 4], 1),
        (6, [3, 4], 1),
        (1, 4, 2),
        (4, 4, 2),
    ],
)
def test_orthonormal_columns(dim, local_dim, r):
    """Columns of the output should be orthonormal."""
    E = entangled_subspace(dim, local_dim, r)
    ld = [local_dim, local_dim] if isinstance(local_dim, int) else local_dim
    assert E.shape == (ld[0] * ld[1], dim)
    # Check orthonormality: E^T E should be identity
    gram = E.conj().T @ E
    np.testing.assert_allclose(gram, np.eye(dim), atol=1e-10)


@pytest.mark.parametrize(
    "dim, local_dim, r",
    [
        (1, 3, 1),
        (4, 3, 1),
        (4, 4, 1),
        (1, 4, 2),
    ],
)
def test_schmidt_rank_exceeds_r(dim, local_dim, r):
    """Every column should have Schmidt rank > r (i.e., be r-entangled)."""
    ld = [local_dim, local_dim] if isinstance(local_dim, int) else local_dim
    E = entangled_subspace(dim, ld, r)
    for col in range(dim):
        vec = E[:, col].reshape(-1, 1)
        sr = schmidt_rank(vec, ld)
        assert sr > r, f"Column {col} has Schmidt rank {sr}, expected > {r}"


def test_max_dimension_3x3():
    """Maximum 1-entangled subspace in 3x3 is (3-1)*(3-1) = 4."""
    E = entangled_subspace(4, 3, 1)
    assert E.shape == (9, 4)


def test_max_dimension_rectangular():
    """Maximum 1-entangled subspace in 3x4 is (3-1)*(4-1) = 6."""
    E = entangled_subspace(6, [3, 4], 1)
    assert E.shape == (12, 6)


def test_exceeds_max_dimension_raises():
    """Requesting too large a subspace should raise ValueError."""
    with pytest.raises(ValueError, match="No 1-entangled subspace"):
        entangled_subspace(5, 3, 1)


def test_exceeds_max_dimension_r2_raises():
    """r=2 in 3x3: max is (3-2)*(3-2)=1, so dim=2 should fail."""
    with pytest.raises(ValueError, match="No 2-entangled subspace"):
        entangled_subspace(2, 3, 2)


def test_single_vector():
    """Single basis vector should be a unit vector."""
    E = entangled_subspace(1, 3, 1)
    assert E.shape == (9, 1)
    assert np.isclose(np.linalg.norm(E[:, 0]), 1.0)


def test_scalar_local_dim():
    """Integer local_dim should be treated as [d, d]."""
    E1 = entangled_subspace(4, 3, 1)
    E2 = entangled_subspace(4, [3, 3], 1)
    assert E1.shape == E2.shape
