"""Test random_state_vector."""
import numpy as np
import pytest

from toqito.rand import random_state_vector
from toqito.state_props import is_pure


@pytest.mark.parametrize("dim", range(2, 8))
@pytest.mark.parametrize("is_real", [True, False])
@pytest.mark.parametrize("k_param", range(0, 1))
def test_random_state_vector(dim, is_real, k_param):
    # We expect the density matrix of any random state vector to be pure.
    vec = random_state_vector(dim=dim, is_real=is_real, k_param=k_param)
    mat = vec @ vec.conj().T
    np.testing.assert_equal(is_pure(mat), True)
