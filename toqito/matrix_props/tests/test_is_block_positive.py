"""Test is_block_positive."""

from unittest.mock import patch

import numpy as np
import pytest
from picos import partial_transpose

from toqito.channels import choi
from toqito.matrix_props import is_block_positive
from toqito.perms import swap, swap_operator
from toqito.states import bell

# matrix for lemma 3 of :cite:`Bandyopadhyay_2015_Limitations`
b_0 = bell(0)
b_3 = bell(3)
v_0 = np.kron(b_0, b_0)
y_mat = (
    np.kron(np.eye(4), b_0 @ b_0.T) / 2 + np.kron(b_3 @ b_3.T, partial_transpose(b_3 @ b_3.T, [0]))
) / 3 - v_0 @ v_0.T / 4
mat = swap(y_mat, [2, 3], [2, 2, 2, 2])


def test_is_block_positive_not_block_positive():
    """Small tolerance for lower bound block positive condition."""
    # Create a matrix that is definitely not block positive
    mat = np.array([[-1, 0], [0, 1]])

    # Set a very small tolerance to ensure the lower bound condition is met
    assert not is_block_positive(mat, rtol=1e-10)


@pytest.mark.parametrize(
    "input_mat, expected_bool_1_block, expected_bool_2_block",
    [
        # Test Swap is 1-block positive but not 2-block positive
        (swap_operator(2), True, False),
        (swap_operator(3), True, False),
        (swap_operator(4), True, False),
        # Test Choi map is 1-block positive but not 2-block positive.
        (choi(), True, False),
        # Test that the positive linear map introduced in :cite:`Bandyopadhyay_2015_Limitations` is block positive
        (mat, True, None),
        # non-hermitian input is not is_block_positive
        (np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), False, None),
    ],
)
def test_is_block_positive(input_mat, expected_bool_1_block, expected_bool_2_block):
    """Test function works as expected for valid default inputs."""
    if expected_bool_2_block is not None:
        if expected_bool_1_block is True:
            assert is_block_positive(input_mat, rtol=0.001)
        else:
            assert not is_block_positive(input_mat, rtol=0.001)
        if expected_bool_2_block is True:
            assert is_block_positive(input_mat, k=2, rtol=0.001)
        else:
            assert not is_block_positive(input_mat, k=2, rtol=0.001)

    # when expected_bool_2_block  is None
    if expected_bool_1_block is True:
        assert is_block_positive(input_mat, rtol=0.001)
    else:
        assert not is_block_positive(input_mat, rtol=0.001)


@pytest.mark.parametrize(
    "input_mat, expected_bool",
    [
        # Test Swap is 1-block positive but not 2-block positive
        (swap_operator(2), True),
        (swap_operator(3), True),
        (swap_operator(4), True),
    ],
)
def test_dim_None(input_mat, expected_bool):
    """Check input dimensions are set correctly when the input is None."""
    if expected_bool is True:
        assert is_block_positive(input_mat, 1, None)


@pytest.mark.parametrize(
    "input_mat, input_dim",
    [
        # Test Swap is 1-block positive but not 2-block positive
        (swap_operator(2), 2),
        ((swap_operator(2), [2, 2])),
    ],
)
def test_dim_input(input_mat, input_dim):
    """Check input dimensions are set correctly when the input dim is an int or list."""
    assert is_block_positive(input_mat, 1, input_dim)
