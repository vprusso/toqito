"""Tests for kraus to channel."""

import numpy as np
import pytest

import toqito.state_ops
from toqito.channel_ops import kraus_to_channel

dim = 2**2
kraus_list = [np.random.randint(-1, 4, (2, dim, dim)) for _ in range(12)]

vector = np.random.randint(-3, 3, (dim, 1))
dm = toqito.matrix_ops.to_density_matrix(vector)
vec_dm = toqito.matrix_ops.vec(dm)

@pytest.mark.parametrize(
    "kraus_list",
    [
        (kraus_list)
    ],
)
def test_kraus_to_channel(kraus_list):
    """Test kraus_tochannel works as expected for valid inputs."""
    calculated = kraus_to_channel(kraus_list)

    value = sum(A @ dm @ B.conj().T for A, B in kraus_list)

    assert toqito.matrix_ops.unvec(calculated @ vec_dm).all() == value.all()
