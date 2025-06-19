"""Test tensor_unravel."""

import numpy as np
import pytest

from toqito.matrix_ops.tensor_unravel import tensor_unravel


def valid_2d_tensor():
    return np.array([[-1, -1], [-1, 1]])

def valid_3d_tensor():
    arr = np.full((2, 2, 2), -1)
    arr[1, 1, 1] = 1
    return arr

def multiple_unique_tensor():
    arr = np.full((2, 2, 2), -1)
    arr[0, 0, 0] = 1
    arr[1, 1, 1] = 1
    return arr

@pytest.mark.parametrize(
    "tensor_input, expected_output, expected_exception",
    [
        # Valid 2D tensor with one +1 at (1,1)
        (valid_2d_tensor, np.array([1, 1, 1]), None),
        # Valid 3D tensor with one +1 at (1,1,1)
        (valid_3d_tensor, np.array([1, 1, 1, 1]), None),
        # Invalid tensor: all values same (no unique)
        (lambda: np.full((2, 2), -1), None, ValueError),
        # Invalid tensor: two +1s (not unique)
        (multiple_unique_tensor, None, ValueError),
    ],
)
def test_tensor_unravel(tensor_input, expected_output, expected_exception):
    """Test unraveling clause tensors into 1D index-value form."""
    tensor = tensor_input() if callable(tensor_input) else tensor_input

    if expected_exception:
        with pytest.raises(expected_exception):
            tensor_unravel(tensor)
    else:
        result = tensor_unravel(tensor)
        assert np.array_equal(result, expected_output)
