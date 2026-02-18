"""Test choi."""

import numpy as np
import pytest

from toqito.channels import choi


def test_choi_standard():
    """The standard Choi map."""
    expected_res = np.array(
        [
            [1, 0, 0, 0, -1, 0, 0, 0, -1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [-1, 0, 0, 0, 1, 0, 0, 0, -1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [-1, 0, 0, 0, -1, 0, 0, 0, 1],
        ]
    )

    res = choi()

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_choi_reduction():
    """The reduction map is the map R defined by: R(X) = Tr(X)I - X.

    The reduction map is the Choi map that arises when a = 0, b = c = 1.
    """
    expected_res = np.array(
        [
            [0, 0, 0, 0, -1, 0, 0, 0, -1],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [-1, 0, 0, 0, 0, 0, 0, 0, -1],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [-1, 0, 0, 0, -1, 0, 0, 0, 0],
        ]
    )

    res = choi(0, 1, 1)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_choi_apply_channel_false():
    """Test that apply_channel=False returns the Choi matrix (existing behavior)."""
    res = choi()
    # Should return a 9x9 Choi matrix
    np.testing.assert_equal(res.shape, (9, 9))


def test_choi_apply_channel_with_input_mat():
    """Test apply_channel=True with a valid input matrix."""
    # Create a 3x3 density matrix for a qutrit state
    test_input_mat = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    res = choi(input_mat=test_input_mat, apply_channel=True)
    # Should return a 3x3 matrix (result of applying the channel)
    np.testing.assert_equal(res.shape, (3, 3))


def test_choi_apply_channel_no_input_mat():
    """Test apply_channel=True with input_mat=None raises ValueError."""
    with pytest.raises(ValueError, match="input_mat is required when apply_channel=True"):
        choi(apply_channel=True)
