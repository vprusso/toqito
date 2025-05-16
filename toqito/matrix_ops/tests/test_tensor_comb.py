"""Test tensor_comb."""

import numpy as np
import pytest

from toqito.matrix_ops import tensor_comb, to_density_matrix


@pytest.mark.parametrize(
    "states, k, mode, density_matrix, expected_keys, expected_error",
    [
        # Injective (default).
        ([np.array([1, 0]), np.array([0, 1])], 3, "injective", False, None, ValueError),
        # Invalid mode.
        ([np.array([1, 0]), np.array([0, 1])], 2, "invalid", False, None, ValueError),
        # Injective mode with k = 1.
        ([np.array([1, 0]), np.array([0, 1])], 1, "injective", False, [(0,), (1,)], None),
        # Non-injective mode with k = 1.
        ([np.array([1, 0]), np.array([0, 1])], 1, "non-injective", False, [(0,), (1,)], None),
        # Non-injective mode with k = 2.
        ([np.array([1, 0]), np.array([0, 1])], 2, "non-injective", False, [(0, 0), (0, 1), (1, 0), (1, 1)], None),
        # Non-injective mode with k = 3.
        (
            [np.array([1, 0]), np.array([0, 1])],
            3,
            "non-injective",
            False,
            [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)],
            None,
        ),
    ],
)
def test_tensor_comb(states, k, mode, density_matrix, expected_keys, expected_error):
    """Test the tensor_comb function."""
    if expected_error:
        with pytest.raises(expected_error) as excinfo:
            tensor_comb(states, k, mode=mode, density_matrix=density_matrix)
            if mode == "injective" and k > len(states):
                assert (
                    str(excinfo.value)
                    == "k must be less than or equal to the number of states for injective sequences."
                )
            if mode not in ("injective", "non-injective", "diagonal"):
                assert str(excinfo.value) == "`mode` must be 'injective', 'non-injective', or 'diagonal'."

    else:
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


def test_tensor_comb_empty_states():
    """Test that tensor_comb raises a ValueError with empty states."""
    with pytest.raises(ValueError, match="Input list of states cannot be empty."):
        tensor_comb([], 2, mode="injective", density_matrix=False)
