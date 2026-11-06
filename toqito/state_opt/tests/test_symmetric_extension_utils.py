"""Test the symmetric-subspace helpers used by the symmetric extension SDPs."""

import cvxpy
import numpy as np
import pytest

from toqito.matrix_ops import partial_trace, partial_transpose
from toqito.perms import symmetric_projection
from toqito.state_opt._symmetric_extension import (
    SymmetricExtension,
    _symmetric_isometry,
    _vec_kron_identity,
    _vec_partial_trace,
    _vec_partial_transpose_perm,
    symmetric_extension_sdp,
)
from toqito.states import bell


@pytest.mark.parametrize("dim, level", [(2, 1), (2, 2), (2, 4), (3, 2), (3, 3), (4, 2)])
def test_symmetric_isometry(dim, level):
    """The isometry has orthonormal columns spanning the symmetric subspace."""
    iso = _symmetric_isometry(dim, level).toarray()
    np.testing.assert_allclose(iso.T @ iso, np.identity(iso.shape[1]), atol=1e-12)
    np.testing.assert_allclose(iso @ iso.T, symmetric_projection(dim, level), atol=1e-12)


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


@pytest.mark.parametrize("dim_x, dim_y, level", [(2, 2, 2), (2, 3, 2), (3, 2, 3)])
def test_symmetric_extension_maps_match_dense_extension(dim_x, dim_y, level):
    """Partial traces and transposes of the extension agree with the dense operator it embeds."""
    extension = SymmetricExtension(dim_x, dim_y, level)
    rng = np.random.default_rng(5)
    n = extension.variable.shape[0]
    mat = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    extension.variable.value = mat + mat.conj().T
    embed = np.kron(np.identity(dim_x), _symmetric_isometry(dim_y, level).toarray())
    sigma = embed @ extension.variable.value @ embed.T
    dims = [dim_x] + [dim_y] * level

    for sys in ([1], list(range(2, level + 1)), list(range(1, level + 1))):
        np.testing.assert_allclose(extension.partial_trace(sys).value, partial_trace(sigma, sys, dims), atol=1e-12)
    for sys in ([1, 2], list(range(level // 2 + 2))):
        np.testing.assert_allclose(
            extension.partial_transpose(sys).value, partial_transpose(sigma, sys, dims), atol=1e-12
        )

    # Transposing X alone is returned on the reduced space, and transposing Y_1 is returned compressed onto
    # X (x) Y_1 (x) Sym^(k-1)(Y); both keep the nonzero spectrum of the transposed extension.
    def nonzero_spectrum(operator):
        spectrum = np.linalg.eigvalsh(operator)
        return np.sort(spectrum[np.abs(spectrum) > 1e-9])

    for sys in ([0], [1]):
        np.testing.assert_allclose(
            nonzero_spectrum(extension.partial_transpose(sys).value),
            nonzero_spectrum(partial_transpose(sigma, sys, dims)),
            atol=1e-9,
        )


@pytest.mark.parametrize("level", [1, 2, 3])
@pytest.mark.parametrize("ppt", [True, False])
def test_symmetric_extension_sdp_shapes(level, ppt):
    """The reduced variable lives on X ⊗ Sym(Y) and the marginal on X ⊗ Y."""
    marginal, constraints = symmetric_extension_sdp(2, 3, level, ppt)
    (z_var,) = marginal.variables()
    sym_dim = symmetric_projection(3, level, partial=True).shape[1]
    assert z_var.shape == (2 * sym_dim, 2 * sym_dim)
    assert marginal.shape == (6, 6)
    assert len(constraints) == 1 + ppt * (1 + (level > 1))


def test_symmetric_extension_sdp_marginal():
    """Lifting a reduced operator through the isometry reproduces the marginal expression."""
    dim_x, dim_y, level = 2, 2, 3
    marginal, _ = symmetric_extension_sdp(dim_x, dim_y, level)
    (z_var,) = marginal.variables()
    rng = np.random.default_rng(2)
    n = z_var.shape[0]
    mat = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    z_var.value = mat + mat.conj().T
    embed = np.kron(np.identity(dim_x), _symmetric_isometry(dim_y, level).toarray())
    sigma = embed @ z_var.value @ embed.T
    np.testing.assert_allclose(marginal.value, partial_trace(sigma, [2, 3], [dim_x] + [dim_y] * level), atol=1e-12)


def test_symmetric_extension_sdp_entangled_infeasible():
    """A Bell state has no PPT symmetric extension, but a mixed state does."""
    for rho, expected in [(bell(0) @ bell(0).conj().T, "infeasible"), (np.identity(4) / 4, "optimal")]:
        marginal, constraints = symmetric_extension_sdp(2, 2, 2)
        problem = cvxpy.Problem(cvxpy.Minimize(0), constraints + [marginal == rho])
        problem.solve()
        assert problem.status.startswith(expected)
