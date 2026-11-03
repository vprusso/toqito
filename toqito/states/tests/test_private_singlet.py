"""Test private_singlet."""

import numpy as np
import pytest

from toqito.matrix_ops import partial_transpose
from toqito.matrix_props import is_density
from toqito.state_props import is_ppt, is_separable
from toqito.states import private_singlet


@pytest.mark.parametrize("shield_dim", [2, 3])
def test_private_singlet_bound_entangled(shield_dim):
    """The state is a density operator on C^{2d} x C^{2d}, PT-invariant across the Alice|Bob cut, and entangled."""
    rho = private_singlet(shield_dim)
    dims = [2 * shield_dim, 2 * shield_dim]
    assert rho.shape == (4 * shield_dim**2, 4 * shield_dim**2)
    assert is_density(rho)
    np.testing.assert_allclose(partial_transpose(rho, [1], dims), rho, atol=1e-12)
    assert is_ppt(rho, dim=dims)
    separable, witness = is_separable(rho, dim=dims)
    assert not separable
    assert "inconclusive" not in witness


def test_private_singlet_invalid():
    """A shield dimension below 2 is rejected."""
    with pytest.raises(ValueError):
        private_singlet(1)
