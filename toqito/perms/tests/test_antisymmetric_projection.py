"""Test antisymmetric_projection."""

import math

import numpy as np
import pytest

from toqito.perms import antisymmetric_projection


@pytest.mark.parametrize(
    "dim, p_param, expected_result",
    [
        # Dimension is 2 and p is equal to 1: identity.
        (2, 1, np.array([[1, 0], [0, 1]])),
        # The `p` value is greater than the dimension `d`: empty antisymmetric subspace.
        (2, 3, np.zeros((8, 8))),
        # The dimension is 2, p is 2: projection onto the (one-dimensional) antisymmetric subspace.
        (2, 2, np.array([[0, 0, 0, 0], [0, 0.5, -0.5, 0], [0, -0.5, 0.5, 0], [0, 0, 0, 0]])),
    ],
)
def test_antisymmetric_projection_value(dim, p_param, expected_result):
    """Test the projector matches the known closed form."""
    proj = antisymmetric_projection(dim=dim, p_param=p_param)
    np.testing.assert_allclose(proj, expected_result, atol=1e-8)


@pytest.mark.parametrize("dim, p_param", [(2, 2), (3, 2), (3, 3), (4, 2), (4, 3)])
def test_antisymmetric_projection_is_projector(dim, p_param):
    """The full projection is a Hermitian idempotent whose rank is the antisymmetric subspace dimension."""
    proj = antisymmetric_projection(dim=dim, p_param=p_param)

    # Idempotent and Hermitian.
    np.testing.assert_allclose(proj @ proj, proj, atol=1e-8)
    np.testing.assert_allclose(proj, proj.conj().T, atol=1e-8)
    # Eigenvalues are 0 or 1 (a genuine orthogonal projection, not its negative).
    eigvals = np.linalg.eigvalsh(proj)
    assert eigvals.min() > -1e-8
    # Rank equals C(dim, p_param), the dimension of the antisymmetric subspace.
    np.testing.assert_allclose(np.trace(proj), math.comb(dim, p_param), atol=1e-8)


@pytest.mark.parametrize("dim, p_param", [(3, 3), (3, 2), (4, 2)])
def test_antisymmetric_projection_partial(dim, p_param):
    """The partial form has orthonormal columns that reconstruct the full projector."""
    basis = antisymmetric_projection(dim=dim, p_param=p_param, partial=True)
    full = antisymmetric_projection(dim=dim, p_param=p_param)

    # Columns are orthonormal.
    np.testing.assert_allclose(basis.conj().T @ basis, np.eye(basis.shape[1]), atol=1e-8)
    # basis @ basis^dagger reconstructs the orthogonal projection.
    np.testing.assert_allclose(basis @ basis.conj().T, full, atol=1e-8)
    # The number of basis vectors equals the antisymmetric subspace dimension.
    assert basis.shape[1] == math.comb(dim, p_param)
