"""Test tensor_comb."""

import numpy as np
import pytest

from toqito.matrix_ops import tensor_comb, to_density_matrix


@pytest.mark.parametrize(
    "states, k, mode, density_matrix, expected_keys",
    [
        # Injective mode with k = 1.
        ([np.array([1, 0]), np.array([0, 1])], 1, "injective", False, [(0,), (1,)]),
        # Non-injective mode with k = 1.
        ([np.array([1, 0]), np.array([0, 1])], 1, "non-injective", False, [(0,), (1,)]),
        # Non-injective mode with k = 2.
        ([np.array([1, 0]), np.array([0, 1])], 2, "non-injective", False, [(0, 0), (0, 1), (1, 0), (1, 1)]),
        # Non-injective mode with k = 3.
        (
            [np.array([1, 0]), np.array([0, 1])],
            3,
            "non-injective",
            False,
            [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)],
        ),
    ],
)
def test_tensor_comb(states, k, mode, density_matrix, expected_keys):
    """Test the tensor_comb function."""
    result = tensor_comb(states, k, mode=mode, density_matrix=density_matrix)
    result_keys = list(result.keys())

    for seq in expected_keys:
        state_seq = [states[i] for i in seq]
        tensor_product = np.array(state_seq[0])
        for state in state_seq[1:]:
            tensor_product = np.kron(tensor_product, state)

        if result[result_keys[0]].ndim == 1:
            np.testing.assert_allclose(result[seq], tensor_product, atol=1e-10)
        else:
            expected_density_matrix = to_density_matrix(tensor_product)
            np.testing.assert_allclose(result[seq], expected_density_matrix, atol=1e-10)


@pytest.mark.parametrize(
    "test_input, test_k, test_density_matrix, test_mode, expected_msg",
    [
        # k is greater than len(states) for injective mode
        (
            [np.array([1, 0]), np.array([0, 1])],
            3,
            False,
            "injective",
            "k must be less than or equal to the number of states for injective sequences.",
        ),
        # invalid mode option
        (
            [np.array([1, 0]), np.array([0, 1])],
            2,
            False,
            "invalid",
            "mode must be injective, non-injective, or diagonal.",
        ),
        # empty input
        ([], 2, False, "injective", "Input list of states cannot be empty."),
    ],
)
def test_raised_errors(test_input, test_k, test_density_matrix, test_mode, expected_msg):
    """Test function raises error as expected for invalid inputs."""
    with pytest.raises(ValueError, match=expected_msg):
        tensor_comb(test_input, test_k, mode=test_mode, density_matrix=test_density_matrix)
