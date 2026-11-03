"""Test divincenzo_upb."""

import numpy as np
import pytest

from toqito.state_props import is_product, is_unextendible_product_basis
from toqito.states import divincenzo_upb, pyramid, tile, upb_state


@pytest.mark.parametrize(
    "angles", [(0.4, 1.1, 0.3, 0.9, 0.7, 2.0), (3 * np.pi / 4, 3 * np.pi / 4, 0.0, 3 * np.pi / 4, 3 * np.pi / 4, 0.0)]
)
def test_divincenzo_upb_orthonormal_product_vectors(angles):
    """Every member consists of five orthonormal product vectors."""
    vecs = divincenzo_upb(*angles)
    assert len(vecs) == 5
    mat = np.column_stack(vecs)
    np.testing.assert_allclose(mat.conj().T @ mat, np.identity(5), atol=1e-12)
    for vec in vecs:
        assert is_product(vec, [3, 3])[0]


def test_divincenzo_upb_is_upb():
    """A generic member is an unextendible product basis."""
    assert is_unextendible_product_basis(divincenzo_upb(0.4, 1.1, 0.3, 0.9, 0.7, 2.0), [3, 3])[0]


def _local_grams(vecs):
    """Moduli of the inner products of the first and second factors of product vectors on C^3 x C^3."""
    factors_a, factors_b = [], []
    for vec in vecs:
        left, sing, right = np.linalg.svd(vec.reshape(3, 3))
        assert sing[1] < 1e-10
        factors_a.append(left[:, 0])
        factors_b.append(right[0])
    factors_a, factors_b = np.array(factors_a), np.array(factors_b)
    return np.abs(factors_a.conj() @ factors_a.T), np.abs(factors_b.conj() @ factors_b.T)


@pytest.mark.parametrize(
    "angle, basis, perm",
    [
        # Tile basis: vectors relabeled by the permutation and the two parties exchanged.
        (3 * np.pi / 4, tile, [0, 1, 3, 2, 4]),
        (np.arccos((np.sqrt(5) - 1) / 2), pyramid, [0, 1, 2, 3, 4]),
    ],
)
def test_divincenzo_upb_tiles_and_pyramid(angle, basis, perm):
    """At the quoted angles the member matches Tile or Pyramid up to local unitaries and a party exchange."""
    gram_a, gram_b = _local_grams(divincenzo_upb(angle, angle, 0, angle, angle, 0))
    ref_a, ref_b = _local_grams([basis(i) for i in range(5)])
    np.testing.assert_allclose(gram_a[np.ix_(perm, perm)], ref_b, atol=1e-9)
    np.testing.assert_allclose(gram_b[np.ix_(perm, perm)], ref_a, atol=1e-9)
    # The two members therefore give states with the same spectrum.
    rho = upb_state(divincenzo_upb(angle, angle, 0, angle, angle, 0))
    assert np.linalg.matrix_rank(rho) == 4


@pytest.mark.parametrize("angles", [(0.0, 1.1, 0.3, 0.9, 0.7, 2.0), (0.4, np.pi / 2, 0.3, 0.9, 0.7, 2.0)])
def test_divincenzo_upb_degenerate_angles(angles):
    """Angles whose cos or sin vanish do not give a UPB and are rejected."""
    with pytest.raises(ValueError):
        divincenzo_upb(*angles)
