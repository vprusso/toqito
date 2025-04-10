"""Tests for random orthonormal basis."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from toqito.matrix_props import is_orthonormal
from toqito.rand import random_orthonormal_basis


@pytest.mark.parametrize("input_dim", range(2, 5))
@pytest.mark.parametrize("bool", [False, True])
def test_random_orth_basis_int_dim(input_dim, bool):
    """Test function works as expected for a valid int input."""
    gen_basis = random_orthonormal_basis(dim=input_dim, is_real=bool)
    assert len(gen_basis) == input_dim
    assert is_orthonormal(np.array(gen_basis))


@pytest.mark.parametrize(
    "dim, is_real, expected",
    [
        (
            2,
            False,
            [
                np.array([-0.172472 + 0.465563j, -0.637147 - 0.589532j]),
                np.array([-0.857827 + 0.132805j, -0.117157 + 0.482462j]),
            ],
        ),
        (2, True, [np.array([-0.26129, -0.96526]), np.array([-0.96526, 0.26129])]),
    ],
)
def test_seed(dim, is_real, expected):
    """Test that the function returns the expected output when seeded."""
    basis = random_orthonormal_basis(dim, is_real, seed=124)
    assert_allclose(basis, expected, rtol=1e-05)
