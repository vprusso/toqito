"""Tests for random orthonormal basis."""

import numpy as np
import pytest

from toqito.matrix_props import is_orthonormal
from toqito.rand import random_orthonormal_basis


@pytest.mark.parametrize("input_dim", range(2, 5))
@pytest.mark.parametrize("bool", [False, True])
def test_random_orth_basis_int_dim(input_dim, bool):
    """Test function works as expected for a valid int input."""
    gen_basis = random_orthonormal_basis(dim=input_dim, is_real = bool)
    assert len(gen_basis) == input_dim
    assert is_orthonormal(np.array(gen_basis))

