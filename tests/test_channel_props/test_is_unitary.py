"""Tests for is_unitary."""
import numpy as np

from toqito.channel_props import is_unitary
from toqito.channels import depolarizing


def test_is_completely_positive_kraus_false():
    """Verify that the identity channel is a unitary channel."""
    kraus_ops = [[np.identity(2), np.identity(2)]]

    np.testing.assert_equal(is_unitary(kraus_ops), True)


def test_is_completely_positive_choi_true():
    """Verify that the Choi matrix of the depolarizing map is not a unitary channel."""
    np.testing.assert_equal(is_unitary(depolarizing(2)), False)


if __name__ == "__main__":
    np.testing.run_module_suite()
    