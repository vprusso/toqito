"""Test gen_tiles2."""

import numpy as np
import pytest

from toqito.state_props import is_ppt, is_product, is_separable
from toqito.states import gen_tiles2, upb_state


def _is_upb(vecs, dim_m, dim_n):
    """Partition test: no product vector is orthogonal to every member."""
    factors_a, factors_b = [], []
    for vec in vecs:
        left, sing, right = np.linalg.svd(vec.reshape(dim_m, dim_n))
        assert sing[1] < 1e-10
        factors_a.append(left[:, 0])
        factors_b.append(right[0])
    factors_a, factors_b = np.array(factors_a), np.array(factors_b)
    num = len(vecs)
    for mask in range(2**num):
        part_a = [i for i in range(num) if mask >> i & 1]
        part_b = [i for i in range(num) if not mask >> i & 1]
        rank_a = np.linalg.matrix_rank(factors_a[part_a]) if part_a else 0
        rank_b = np.linalg.matrix_rank(factors_b[part_b]) if part_b else 0
        if rank_a < dim_m and rank_b < dim_n:
            return False
    return True


@pytest.mark.parametrize("dim_m, dim_n", [(3, 4), (3, 5), (4, 4), (4, 5)])
def test_gen_tiles2_orthonormal_product_vectors(dim_m, dim_n):
    """GenTiles2 has mn - 2m + 1 orthonormal product vectors."""
    vecs = gen_tiles2(dim_m, dim_n)
    assert len(vecs) == dim_m * dim_n - 2 * dim_m + 1
    mat = np.column_stack(vecs)
    np.testing.assert_allclose(mat.conj().T @ mat, np.identity(len(vecs)), atol=1e-12)
    for vec in vecs:
        assert is_product(vec, [dim_m, dim_n])[0]


@pytest.mark.parametrize("dim_m, dim_n", [(3, 4), (3, 5), (4, 4)])
def test_gen_tiles2_is_upb(dim_m, dim_n):
    """The vectors form an unextendible product basis, including for m = 3 and n = 5."""
    assert _is_upb(gen_tiles2(dim_m, dim_n), dim_m, dim_n)


@pytest.mark.parametrize("dim_m, dim_n", [(3, 4), (3, 5), (4, 4)])
def test_gen_tiles2_state(dim_m, dim_n):
    """The associated state has rank 2m - 1, is PPT, and is certified entangled."""
    rho = upb_state(gen_tiles2(dim_m, dim_n))
    assert np.linalg.matrix_rank(rho) == 2 * dim_m - 1
    assert is_ppt(rho, dim=[dim_m, dim_n])
    separable, witness = is_separable(rho, dim=[dim_m, dim_n])
    assert not separable
    assert "inconclusive" not in witness


@pytest.mark.parametrize("dim_m, dim_n", [(3, 3), (2, 4), (4, 3)])
def test_gen_tiles2_invalid_dims(dim_m, dim_n):
    """Dimensions outside n > 3, m >= 3, n >= m are rejected."""
    with pytest.raises(ValueError):
        gen_tiles2(dim_m, dim_n)
