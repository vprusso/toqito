"""Test reduction."""

import numpy as np
import pytest

from toqito.channels import reduction


def test_reduction_map():
    """Test for the standard reduction map."""
    res = reduction(3)
    np.testing.assert_equal(res[4, 0], -1)
    np.testing.assert_equal(res[8, 0], -1)
    np.testing.assert_equal(res[1, 1], 1)
    np.testing.assert_equal(res[2, 2], 1)
    np.testing.assert_equal(res[3, 3], 1)
    np.testing.assert_equal(res[0, 4], -1)
    np.testing.assert_equal(res[8, 4], -1)
    np.testing.assert_equal(res[5, 5], 1)
    np.testing.assert_equal(res[6, 6], 1)
    np.testing.assert_equal(res[7, 7], 1)
    np.testing.assert_equal(res[0, 8], -1)
    np.testing.assert_equal(res[4, 8], -1)


def test_reduction_map_dim_3_k_2():
    """Test for the reduction map with dimension 3 and parameter k = 2."""
    res = reduction(3, 2)
    expected_res = np.array(
        [
            [1, 0, 0, 0, -1, 0, 0, 0, -1],
            [0, 2, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 2, 0, 0, 0, 0, 0],
            [-1, 0, 0, 0, 1, 0, 0, 0, -1],
            [0, 0, 0, 0, 0, 2, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 2, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 2, 0],
            [-1, 0, 0, 0, -1, 0, 0, 0, 1],
        ]
    )

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_reduction_apply_channel_false():
    """Test that apply_channel=False returns the Choi matrix (existing behavior)."""
    res = reduction(3)
    # Should return a 9x9 Choi matrix for dim=3
    np.testing.assert_equal(res.shape, (9, 9))


def test_reduction_apply_channel_with_input_mat():
    """Test apply_channel=True with a valid input matrix."""
    test_input_mat = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    res = reduction(3, input_mat=test_input_mat, apply_channel=True)
    # Should return a 3x3 matrix (result of applying the channel)
    np.testing.assert_equal(res.shape, (3, 3))


def test_reduction_apply_channel_no_input_mat():
    """Test apply_channel=True with input_mat=None raises ValueError."""
    with pytest.raises(ValueError, match="input_mat is required when apply_channel=True"):
        reduction(3, apply_channel=True)
