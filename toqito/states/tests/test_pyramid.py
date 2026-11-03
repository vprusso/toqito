"""Test pyramid."""

import numpy as np
import pytest

from toqito.state_props import is_product, is_unextendible_product_basis
from toqito.states import pyramid


def test_pyramid_orthonormal_product_vectors():
    """The five Pyramid vectors are orthonormal product vectors."""
    vecs = [pyramid(i) for i in range(5)]
    mat = np.column_stack(vecs)
    np.testing.assert_allclose(mat.T @ mat, np.identity(5), atol=1e-12)
    for vec in vecs:
        assert is_product(vec, [3, 3])[0]


def test_pyramid_orthogonality_structure():
    """Indices differing by two are orthogonal on the first factor, by one on the second."""
    height = np.sqrt(1 + np.sqrt(5)) / 2

    def vertex(j):
        vec = np.array([np.cos(2 * np.pi * j / 5), np.sin(2 * np.pi * j / 5), height])
        return vec / np.linalg.norm(vec)

    for j in range(5):
        np.testing.assert_allclose(vertex(j) @ vertex((j + 2) % 5), 0, atol=1e-12)
        np.testing.assert_allclose(pyramid(j), np.kron(vertex(j), vertex((2 * j) % 5)).reshape(-1, 1))


def test_pyramid_is_upb():
    """The Pyramid vectors form an unextendible product basis."""
    assert is_unextendible_product_basis([pyramid(i) for i in range(5)], [3, 3])[0]


@pytest.mark.parametrize("idx", [-1, 5])
def test_pyramid_invalid(idx):
    """Indices outside 0..4 are rejected."""
    with pytest.raises(ValueError):
        pyramid(idx)
