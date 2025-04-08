"""Test tensor_comb for different cases"""
import sys
import numpy as np
import pytest

from toqito.matrix_ops import tensor_comb, to_density_matrix
from toqito.states import basis


@pytest.mark.parametrize(
    "k, states, mode, density_matrix, expected_keys, expected_error",
    [
        (3,
         [np.array([1, 0]), np.array([0, 1])],
         "injective", False,
         None,
         ValueError),
        # Invalid mode
        (2,
         [np.array([1, 0]), np.array([0, 1])],
         "invalid", False,
         None,
         ValueError),
        # Non-injective mode with k = 1
        (1,
         [np.array([1, 0]), np.array([0, 1])],
         "non-injective", False,
         [(0,), (1,)],
         None),
        # Injective mode with k = 1
        (1,
         [np.array([1, 0]), np.array([0, 1])],
         "injective", False,
         [(0,), (1,)],
         None),
        # Non-injective mode with k=2
        (2,
         [np.array([1, 0]), np.array([0, 1])],
         "non-injective", False,
         [(0, 0), (0, 1), (1, 0), (1, 1)],
         None),
    ],
)
def test_tensor_comb(k, states, mode, density_matrix, expected_keys, expected_error):
    """Test the tensor_comb function."""
    if expected_error:
        with pytest.raises(expected_error):
            tensor_comb(k, states, mode=mode, density_matrix=density_matrix)

    else:
        result = tensor_comb(k, states, mode=mode,
                             density_matrix=density_matrix)
        result_keys = list(result.keys())
        # check expected keys
        assert sorted(result.keys()) == sorted(
            expected_keys)  # , f"Expected: {
#            expected_keys}, Got: {list(result.keys())}"

        for seq in expected_keys:
            state_seq = [states[i] for i in seq]
            tensor_product = np.array(state_seq[0])
            for state in state_seq[1:]:
                tensor_product = np.kron(tensor_product, state)

            if result[result_keys[0]].ndim == 1:
                np.testing.assert_allclose(
                    result[seq], tensor_product, atol=1e-10)
            else:
                expected_density_matrix = to_density_matrix(tensor_product)
                np.testing.assert_allclose(
                    result[seq], expected_density_matrix, atol=1e-10)
