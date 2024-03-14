"""Test dephasing."""

import numpy as np

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
