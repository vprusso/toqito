"""Test yu_oh."""

import numpy as np
import pytest

from toqito.matrix_ops import partial_transpose
from toqito.matrix_props import is_density
from toqito.state_props import is_ppt, is_separable
from toqito.states import yu_oh


@pytest.mark.parametrize("dim, x_param, y_param", [(3, 0.5, 0.1), (3, 0.2, 0.3), (4, 0.3, 0.2)])
def test_yu_oh_bound_entangled(dim, x_param, y_param):
    """The state is a PPT density operator, invariant under partial transposition, and certified entangled."""
    rho = yu_oh(dim, x_param, y_param)
    assert rho.shape == (dim**2, dim**2)
    assert is_density(rho)
    np.testing.assert_allclose(partial_transpose(rho, [1], [dim, dim]), rho, atol=1e-12)
    assert is_ppt(rho, dim=[dim, dim])
    separable, witness = is_separable(rho, dim=[dim, dim])
    assert not separable
    assert "inconclusive" not in witness


def test_yu_oh_rank():
    """The two-qutrit member has rank 1 + 1 + 2 = 4: Phi, one antisymmetric vector, and two psi_k."""
    assert np.linalg.matrix_rank(yu_oh(3, 0.5, 0.1)) == 4


@pytest.mark.parametrize(
    "dim, x_param, y_param",
    [
        # Local dimension too small.
        (2, 0.1, 0.1),
        # x^2 + y^2 > 1.
        (3, 0.8, 0.8),
        # delta <= 0.
        (3, 0.7, 0.7),
    ],
)
def test_yu_oh_invalid(dim, x_param, y_param):
    """Invalid dimensions and parameters are rejected."""
    with pytest.raises(ValueError):
        yu_oh(dim, x_param, y_param)
