import pytest
import numpy as np
from toqito.matrix_ops import to_density_matrix, tensor_comb


@pytest.mark.parametrize(
    "k, states, mode, density_matrix, expected_keys, expected_error",
    [
        # Injective mode: k > len(states) (should raise ValueError)
        (3, [np.array([1, 0]), np.array([0, 1])], "injective", True, None, ValueError),

        # Invalid mode (should raise ValueError)
        (2, [np.array([1, 0]), np.array([0, 1])], "invalid", True, None, ValueError),

        # Non-injective mode with k = 1
        (1, [np.array([1, 0]), np.array([0, 1])], "non-injective", True, [(0,), (1,)], None),

        # Injective mode with k = 1
        (1, [np.array([1, 0]), np.array([0, 1])], "injective", True, [(0,), (1,)], None),

        # Diagonal mode with k = 2
        (2, [np.array([1, 0]), np.array([0, 1])], "diagonal", True, [(0, 0), (1, 1)], None),

        # Non-injective mode with k = 2 (density matrix)
        (2, [np.array([1, 0]), np.array([0, 1])], "non-injective", True, [(0, 0), (0, 1), (1, 0), (1, 1)], None),

        # Non-injective mode with k = 2 (without density matrix)
        (2, [np.array([1, 0]), np.array([0, 1])], "non-injective", False, [(0, 0), (0, 1), (1, 0), (1, 1)], None),
    ],
)
def test_tensor_comb(k, states, mode, density_matrix, expected_keys, expected_error):
    """Test the tensor_comb function with various modes and density matrix options."""
    if expected_error:
        with pytest.raises(expected_error):
            tensor_comb(k, states, mode=mode, density_matrix=density_matrix)
    else:
        result = tensor_comb(k, states, mode=mode, density_matrix=density_matrix)
        
        # Check expected keys
        assert sorted(result.keys()) == sorted(expected_keys), f"Expected: {expected_keys}, Got: {list(result.keys())}"
        
        # Verify tensor products
        for seq in expected_keys:
            tensor_product = np.array(states[seq[0]])
            for state in seq[1:]:
                tensor_product = np.kron(tensor_product, state)

            expected_value = to_density_matrix(tensor_product) if density_matrix else tensor_product
            np.testing.assert_array_almost_equal(
                result[seq], expected_value, err_msg=f"Mismatch for sequence {seq}"
            )
