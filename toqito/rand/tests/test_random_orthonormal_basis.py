"""Tests for random orthonormal basis."""

import numpy as np
import pytest
from numpy.ma.testutils import assert_array_almost_equal

from toqito.matrix_props import is_orthonormal
from toqito.rand import random_orthonormal_basis


@pytest.mark.parametrize("input_dim", range(2, 5))
@pytest.mark.parametrize("bool", [False, True])
def test_random_orth_basis_int_dim(input_dim, bool):
    """Test function works as expected for a valid int input."""
    gen_basis = random_orthonormal_basis(dim=input_dim, is_real = bool)
    assert len(gen_basis) == input_dim
    assert is_orthonormal(np.array(gen_basis))

@pytest.mark.parametrize(
    "dim, is_real, expected",
    [
        (
            2,
            False,
            [
                np.array([0.51345691+0.6924564j, 0.16581665-0.47892689j]),
                np.array([0.38709015+0.32715034j, -0.08796944+0.85755189j])
            ]
        ),
(
            2,
            True,
            [
                np.array([0.95160818, 0.30731397]),
                np.array([-0.30731397, 0.95160818])
            ]
        ),
    ]
)
def test_seed(dim, is_real, expected):
    """Test that the function returns the expected output when seeded."""
    basis = random_orthonormal_basis(dim, is_real, seed=123)
    assert_array_almost_equal(basis, expected)
