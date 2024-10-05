"""Test tensor_comb."""

import numpy as np
import pytest

from toqito.matrix_ops import tensor_comb
from toqito.states import basis

e_0, e_1 = basis(2, 0), basis(2, 1)


@pytest.mark.parametrize(
    "states, k, expected",
    [
        # Test tensor_comb for sequence length 1
        ([e_0, e_1], 1, {(0,): np.outer(e_0, e_0), (1,): np.outer(e_1, e_1)}),

        # Test tensor_comb for sequence length 2
        ([e_0, e_1], 2, {
            (0, 0): np.outer(np.kron(e_0, e_0), np.kron(e_0, e_0)),
            (0, 1): np.outer(np.kron(e_0, e_1), np.kron(e_0, e_1)),
            (1, 0): np.outer(np.kron(e_1, e_0), np.kron(e_1, e_0)),
            (1, 1): np.outer(np.kron(e_1, e_1), np.kron(e_1, e_1)),
        }),

        # Test tensor_comb for sequence length 3
        ([e_0, e_1], 3, {
            (0, 0, 0): np.outer(np.kron(np.kron(e_0, e_0), e_0), np.kron(np.kron(e_0, e_0), e_0)),
            (0, 0, 1): np.outer(np.kron(np.kron(e_0, e_0), e_1), np.kron(np.kron(e_0, e_0), e_1)),
            # other possible sequences omitted for brevity...
        })
    ]
)
def test_tensor_comb(states, k, expected):
    """Test tensor_comb for various sequence lengths."""
    result = tensor_comb(states, k)
    for key in expected:
        assert np.allclose(result[key], expected[key])


def test_tensor_comb_empty_states():
    """Test that tensor_comb raises a ValueError with empty states."""
    with pytest.raises(ValueError, match="Input list of states cannot be empty."):
        tensor_comb([], 2)
