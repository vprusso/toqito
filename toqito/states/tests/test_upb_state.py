"""Test upb_state."""

import numpy as np
import pytest

from toqito.matrix_ops import partial_transpose
from toqito.matrix_props import is_density
from toqito.state_props import is_ppt
from toqito.states import pyramid, tile, upb_state


@pytest.mark.parametrize("basis", [tile, pyramid])
def test_upb_state_tile_and_pyramid(basis):
    """The state on the complement of a 3x3 UPB has rank 4, is PPT, and annihilates the basis."""
    vecs = [basis(i) for i in range(5)]
    rho = upb_state(vecs)
    assert is_density(rho)
    assert np.linalg.matrix_rank(rho) == 4
    np.testing.assert_allclose(partial_transpose(rho, [1], [3, 3]), rho, atol=1e-12)
    assert is_ppt(rho, dim=[3, 3])
    for vec in vecs:
        np.testing.assert_allclose(rho @ vec, 0, atol=1e-12)


def test_upb_state_is_projector():
    """The output is the normalized projector onto the orthogonal complement."""
    vecs = [tile(i) for i in range(5)]
    rho = upb_state(vecs)
    np.testing.assert_allclose(4 * rho @ rho, rho, atol=1e-12)


def test_upb_state_not_orthonormal():
    """Non-orthonormal input is rejected."""
    with pytest.raises(ValueError, match="orthonormal"):
        upb_state([tile(0), tile(0)])


def test_upb_state_spans_space():
    """A complete basis has an empty complement."""
    with pytest.raises(ValueError, match="whole space"):
        upb_state([np.identity(3)[:, [i]] for i in range(3)])
