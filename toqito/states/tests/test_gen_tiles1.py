"""Test gen_tiles1."""

import numpy as np
import pytest

from toqito.state_props import is_ppt, is_product
from toqito.states import gen_tiles1, upb_state


@pytest.mark.parametrize("dim", [4, 6])
def test_gen_tiles1_orthonormal_product_vectors(dim):
    """GenTiles1 has d^2 - 2d + 1 orthonormal product vectors."""
    vecs = gen_tiles1(dim)
    assert len(vecs) == dim**2 - 2 * dim + 1
    mat = np.column_stack(vecs)
    np.testing.assert_allclose(mat.conj().T @ mat, np.identity(len(vecs)), atol=1e-12)
    for vec in vecs:
        assert is_product(vec, [dim, dim])[0]


@pytest.mark.parametrize("dim", [4, 6])
def test_gen_tiles1_state(dim):
    """The associated state has rank 2d - 1 and is PPT."""
    rho = upb_state(gen_tiles1(dim))
    assert np.linalg.matrix_rank(rho) == 2 * dim - 1
    assert is_ppt(rho, dim=[dim, dim])


def test_gen_tiles1_no_product_vector_in_complement():
    """No product vector is orthogonal to all GenTiles1 vectors for d = 4 (the UPB property)."""
    vecs = gen_tiles1(4)
    factors_a, factors_b = [], []
    for vec in vecs:
        left, sing, right = np.linalg.svd(vec.reshape(4, 4))
        assert sing[1] < 1e-10
        factors_a.append(left[:, 0])
        factors_b.append(right[0].conj())
    factors_a, factors_b = np.array(factors_a), np.array(factors_b)
    # A product vector orthogonal to every member exists iff some split of the members leaves
    # both the first factors and the second factors rank deficient.
    num = len(vecs)
    for mask in range(2**num):
        part_a = [i for i in range(num) if mask >> i & 1]
        part_b = [i for i in range(num) if not mask >> i & 1]
        rank_a = np.linalg.matrix_rank(factors_a[part_a]) if part_a else 0
        rank_b = np.linalg.matrix_rank(factors_b[part_b]) if part_b else 0
        assert rank_a == 4 or rank_b == 4


@pytest.mark.parametrize("dim", [2, 3, 5])
def test_gen_tiles1_invalid_dim(dim):
    """Odd dimensions and dimensions below 4 are rejected."""
    with pytest.raises(ValueError):
        gen_tiles1(dim)
