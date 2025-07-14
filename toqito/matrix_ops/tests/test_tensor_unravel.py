"""Test tensor_unravel."""

import numpy as np
import pytest

from toqito.matrix_ops import tensor_unravel


@pytest.mark.parametrize(
    "tensor_input, expected_output, expected_exception",
    [
        # Valid 2D tensor with one +1 at (1, 1).
        (np.array([[-1, -1], [-1, 1]]), np.array([1, 1, 1]), None),
        # Valid 3D tensor with one +1 at (1, 1, 1).
        (np.array([[[-1, -1], [-1, -1]], [[-1, -1], [-1, 1]]]), np.array([1, 1, 1, 1]), None),
        # Invalid tensor: all values are the same (no unique).
        (np.full((2, 2), -1), None, ValueError),
        # Invalid tensor: all values are the same (no unique).
        (np.full((2, 2), -1), None, ValueError),
        # Invalid tensor: two +1s (not unique).
        (np.array([[np.nan, -1], [-1, -1]]), None, ValueError),
        # Invalid tensor: not unique.
        (np.array([[-1, -1], [1, 1]]), None, ValueError),
    ],
)
def test_tensor_unravel(tensor_input, expected_output, expected_exception):
    """Test unraveling clause tensors into 1D index-value form."""
    tensor = tensor_input

    if expected_exception:
        with pytest.raises(expected_exception):
            tensor_unravel(tensor)
    else:
        result = tensor_unravel(tensor)
        assert np.array_equal(result, expected_output)
