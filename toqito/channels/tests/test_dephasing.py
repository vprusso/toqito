"""Test dephasing."""

import numpy as np
import pytest

from toqito.channel_ops import apply_channel
from toqito.channels import dephasing


def test_dephasing_completely_dephasing():
    """The completely dephasing channel kills everything off diagonal."""
    test_input_mat = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])

    expected_res = np.array([[1, 0, 0, 0], [0, 6, 0, 0], [0, 0, 11, 0], [0, 0, 0, 16]])

    res = apply_channel(test_input_mat, dephasing(4))

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_dephasing_partially_dephasing():
    """The partially dephasing channel for `p = 0.5`."""
    test_input_mat = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    param_p = 0.5

    res = apply_channel(test_input_mat, dephasing(4, param_p))
    expected_res = (1 - param_p) * np.diag(np.diag(test_input_mat)) + param_p * test_input_mat

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_dephasing_apply_channel_with_input_mat():
    """Test apply_channel=True with a valid input matrix."""
    test_input_mat = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    res = dephasing(4, input_mat=test_input_mat, apply_channel=True)
    expected_res = np.array([[1, 0, 0, 0], [0, 6, 0, 0], [0, 0, 11, 0], [0, 0, 0, 16]])
    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_dephasing_apply_channel_no_input_mat():
    """Test apply_channel=True with input_mat=None raises ValueError."""
    with pytest.raises(ValueError, match="input_mat is required when apply_channel=True"):
        dephasing(4, apply_channel=True)


def test_dephasing_return_kraus():
    """Test apply_channel=False with return_kraus=True returns Kraus operators."""
    kraus_ops = dephasing(2, return_kraus=True)
    # Should return a list of Kraus operators
    assert isinstance(kraus_ops, list)
    # Each Kraus operator should be a 2x2 matrix
    for kraus in kraus_ops:
        np.testing.assert_equal(kraus.shape, (2, 2))
