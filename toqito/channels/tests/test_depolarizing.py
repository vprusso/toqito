"""Test depolarizing."""

import re

import numpy as np
import pytest

from toqito.channel_ops import apply_channel
from toqito.channels import depolarizing


def test_depolarizing_complete_depolarizing():
    """Maps every density matrix to the maximally-mixed state."""
    test_input_mat = np.array([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]])

    expected_res = 1 / 4 * np.identity(4)

    res = apply_channel(test_input_mat, depolarizing(4))

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_depolarizing_partially_depolarizing():
    """The partially depolarizing channel for `p = 0.5`."""
    test_input_mat = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    param_p = 0.5

    res = apply_channel(test_input_mat, depolarizing(4, param_p))
    expected_res = (1 - param_p) * np.trace(test_input_mat) * np.identity(4) / 4 + param_p * test_input_mat

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


@pytest.mark.parametrize("param_p", [-0.1, 1.5])
def test_depolarizing_invalid_param_p(param_p):
    """Test invalid input to param_p."""
    with pytest.raises(ValueError, match=re.escape("The depolarizing probability must be between 0 and 1.")):
        depolarizing(2, param_p)
