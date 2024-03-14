"""Tests for is_quantum_channel."""

import numpy as np

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
