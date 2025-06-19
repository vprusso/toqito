"""Test tensor_unravel."""

import numpy as np
import pytest

from toqito.matrix_ops.tensor_unravel import tensor_unravel


@pytest.mark.parametrize(
    "tensor_input, expected_output, expected_exception",
    [
        # Valid 2D tensor with one +1 at (1,1)
        (np.array([[-1, -1], [-1, 1]]), np.array([1, 1, 1]), None),
        # Valid 3D tensor with one +1 at (1,1,1)
        (np.array(
            [[[-1, -1], [-1, -1]],
             [[-1, -1], [-1, 1]]]
        ), np.array([1, 1, 1, 1]), None),
        # Invalid tensor: all values same (no unique)
        (np.full((2, 2), -1), None, ValueError),
<<<<<<< HEAD
        (valid_2d_tensor, np.array([1, 1, 1]), None),
        # Valid 3D tensor with one +1 at (1,1,1)
        (np.array(
            [[[-1, -1], [-1, -1]],
             [[-1, -1], [-1, 1]]]
        ), np.array([1, 1, 1, 1]), None),
        # Invalid tensor: all values same (no unique)
        (np.full((2, 2), -1), None, ValueError),
=======
>>>>>>> 50aec3b8 (Modifield test_tensor_unravel.py)
        # Invalid tensor: two +1s (not unique)
        (np.array(
            [[[-1, -1], [-1, -1]],
             [[ 1, -1], [-1,  1]]]
        ), None, ValueError),
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
