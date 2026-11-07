"""Test the sparse vectorized maps behind the cvxpy partial trace and partial transpose."""

import numpy as np
import pytest

from toqito.matrix_ops import partial_trace, partial_transpose
from toqito.matrix_ops._vec_maps import _vec_kron_identity, _vec_partial_trace, _vec_partial_transpose_perm


@pytest.mark.parametrize("sys", [[0], [1], [2, 3], [0, 2]])
def test_vec_partial_trace(sys):
    """The vectorized partial trace agrees with `partial_trace`."""
    dims = [2, 3, 2, 3]
    rng = np.random.default_rng(0)
    mat = rng.normal(size=(36, 36)) + 1j * rng.normal(size=(36, 36))
    vec_map = _vec_partial_trace(dims, sys)
    out_dim = int(np.sqrt(vec_map.shape[0]))
    np.testing.assert_allclose((vec_map @ mat.ravel()).reshape(out_dim, out_dim), partial_trace(mat, sys, dims))


@pytest.mark.parametrize("sys", [0, 1, 2, 3])
def test_vec_partial_transpose_perm(sys):
    """The vectorized partial transpose agrees with `partial_transpose`."""
    dims = [2, 3, 2, 3]
    rng = np.random.default_rng(1)
    mat = rng.normal(size=(36, 36)) + 1j * rng.normal(size=(36, 36))
    perm = _vec_partial_transpose_perm(dims, sys)
    np.testing.assert_allclose(mat.ravel()[perm].reshape(36, 36), partial_transpose(mat, [sys], dims))


@pytest.mark.parametrize("sys", [[0, 1], [1, 3], [0, 2, 3]])
def test_vec_partial_transpose_perm_several_systems(sys):
    """Transposing several subsystems at once agrees with `partial_transpose`."""
    dims = [2, 3, 2, 3]
    rng = np.random.default_rng(3)
    mat = rng.normal(size=(36, 36)) + 1j * rng.normal(size=(36, 36))
    perm = _vec_partial_transpose_perm(dims, sys)
    np.testing.assert_allclose(mat.ravel()[perm].reshape(36, 36), partial_transpose(mat, sys, dims))


@pytest.mark.parametrize("dim, dim_identity", [(2, 2), (3, 2), (2, 4)])
def test_vec_kron_identity(dim, dim_identity):
    """The vectorized map reproduces tensoring with an identity on the right."""
    rng = np.random.default_rng(4)
    mat = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    out = (_vec_kron_identity(dim, dim_identity) @ mat.ravel()).reshape(dim * dim_identity, dim * dim_identity)
    np.testing.assert_allclose(out, np.kron(mat, np.identity(dim_identity)))
