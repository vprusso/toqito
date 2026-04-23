"""Test has_symmetric_inner_extension."""

import numpy as np
import pytest

from toqito.state_props import has_symmetric_inner_extension
from toqito.states import max_entangled, max_mixed


def test_has_symmetric_inner_extension_qetlab_example():
    """QETLAB example: a specific 3x3 PPT inner extension certifies separability."""
    rho = np.array(
        [
            [11, 4, -1, -1, -1, -1, -1, -1, -1],
            [4, 11, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, 11, -1, -1, 4, -1, -1, -1],
            [-1, -1, -1, 11, -1, -1, 4, -1, -1],
            [-1, -1, -1, -1, 16, -1, -1, -1, -1],
            [-1, -1, 4, -1, -1, 11, -1, -1, -1],
            [-1, -1, -1, 4, -1, -1, 11, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, 11, 4],
            [-1, -1, -1, -1, -1, -1, -1, 4, 11],
        ],
        dtype=float,
    )

    assert has_symmetric_inner_extension(rho, level=2, dim=[3, 3], ppt=True)


def test_has_symmetric_inner_extension_rejects_maximally_entangled_qutrit():
    """A maximally entangled qutrit state is not certified by the inner cone."""
    psi = max_entangled(3)
    rho = psi @ psi.conj().T

    assert not has_symmetric_inner_extension(rho, level=2, dim=[3, 3], ppt=True)


def test_has_symmetric_inner_extension_level_must_be_at_least_two():
    """`level` below 2 is rejected with a clear error."""
    rho = np.eye(4) / 4
    with pytest.raises(ValueError, match="level"):
        has_symmetric_inner_extension(rho, level=1)


def test_has_symmetric_inner_extension_default_dim_inferred():
    """With `dim=None` the function infers a balanced bipartite split."""
    rho = max_mixed(4, is_sparse=False)
    assert has_symmetric_inner_extension(rho)


def test_has_symmetric_inner_extension_scalar_dim():
    """A scalar integer `dim` should be accepted when it evenly divides the matrix size."""
    rho = max_mixed(4, is_sparse=False)
    assert has_symmetric_inner_extension(rho, level=2, dim=2, ppt=True)


def test_has_symmetric_inner_extension_scalar_dim_must_divide():
    """A scalar `dim` that does not evenly divide the matrix size is rejected."""
    rho = np.eye(4) / 4
    with pytest.raises(ValueError, match="evenly divide"):
        has_symmetric_inner_extension(rho, level=2, dim=3)


def test_has_symmetric_inner_extension_trivial_subsystem_shortcut():
    """When one subsystem has dimension 1, the problem reduces to positive semidefiniteness."""
    rho_pos = np.eye(4) / 4
    assert has_symmetric_inner_extension(rho_pos, level=2, dim=[1, 4])

    rho_not_psd = np.array([[1.0, 0.0], [0.0, -1.0]])
    assert not has_symmetric_inner_extension(rho_not_psd, level=2, dim=[1, 2])


def test_has_symmetric_inner_extension_two_qubit_jacobi_shortcut():
    """`dim_y == 2` makes the Jacobi degree zero so `eta = 1.0` is taken directly."""
    rho = max_mixed(4, is_sparse=False)
    assert has_symmetric_inner_extension(rho, level=2, dim=[2, 2], ppt=True)


def test_has_symmetric_inner_extension_non_ppt_variant():
    """The `ppt=False` path exercises the non-PPT affine image and skips the PT constraint."""
    rho = max_mixed(4, is_sparse=False)
    assert has_symmetric_inner_extension(rho, level=2, dim=[2, 2], ppt=False)
