"""Tests for is_quantum_channel."""

import re

import numpy as np
import pytest

from toqito.channel_props import is_quantum_channel
from toqito.channels import depolarizing


def test_is_completely_positive_kraus_false():
    """Verify non-completely positive channel as Kraus ops as False."""
    unitary_mat = np.array([[1, 1], [-1, 1]]) / np.sqrt(2)
    kraus_ops = [[np.identity(2), np.identity(2)], [unitary_mat, -unitary_mat]]

    np.testing.assert_equal(is_quantum_channel(kraus_ops), False)


def test_is_completely_positive_choi_true():
    """Verify Choi matrix of the depolarizing map as a quantum channel."""
    np.testing.assert_equal(is_quantum_channel(depolarizing(2)), True)


def test_is_quantum_channel_invalid_input_returns_false():
    """Ensure is_quantum_channel returns False when given invalid input."""
    bad_input = np.eye(3)  # not a matrix of dim = perfect square.
    result = is_quantum_channel(bad_input)
    assert result is False


@pytest.mark.parametrize(
    "bad_phi",
    [
        # not ndarray or list.
        123,
        # list but not list of lists.
        [np.eye(2)],
        # wrong inner type.
        [["not a matrix"]],
        # mixed types.
        [[np.eye(2), "bad"]],
    ],
)
def test_is_quantum_channel_invalid_types_raise(bad_phi):
    """Testing invalid input types."""
    with pytest.raises(
        TypeError,
        match=re.escape(
            "phi must be either a numpy array (Choi matrix) or a list of lists of numpy arrays (Kraus operators)."
        ),
    ):
        is_quantum_channel(bad_phi)
