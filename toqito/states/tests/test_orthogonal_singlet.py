"""Test orthogonal_singlet."""

import numpy as np
import pytest

from toqito.matrix_ops import partial_transpose
from toqito.matrix_props import is_density
from toqito.state_props import is_ppt, is_separable
from toqito.states import orthogonal_singlet


@pytest.mark.parametrize("shield_dim", [2, 3, 4])
def test_orthogonal_singlet_bound_entangled(shield_dim):
    """The state is a PPT density operator on C^{2d} x C^{2d} that is certified entangled."""
    rho = orthogonal_singlet(shield_dim)
    dims = [2 * shield_dim, 2 * shield_dim]
    assert rho.shape == (4 * shield_dim**2, 4 * shield_dim**2)
    assert is_density(rho)
    assert is_ppt(rho, dim=dims)
    separable, witness = is_separable(rho, dim=dims)
    assert not separable
    assert "inconclusive" not in witness


def test_orthogonal_singlet_not_pt_invariant():
    """Unlike the private singlet, the state is not invariant under partial transposition."""
    rho = orthogonal_singlet(2)
    assert not np.allclose(partial_transpose(rho, [1], [4, 4]), rho)


@pytest.mark.parametrize("shield_dim", [1, 5, 6])
def test_orthogonal_singlet_invalid(shield_dim):
    """Shield dimensions other than 3 or a power of 2 are rejected."""
    with pytest.raises(ValueError):
        orthogonal_singlet(shield_dim)
