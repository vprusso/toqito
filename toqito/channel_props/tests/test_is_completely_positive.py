"""Tests for is_completely_positive."""
import numpy as np

from toqito.channel_props import is_completely_positive
from toqito.channels import depolarizing


def test_is_completely_positive_kraus_false():
    """Verify non-completely positive channel as Kraus ops as False."""
    unitary_mat = np.array([[1, 1], [-1, 1]]) / np.sqrt(2)
    kraus_ops = [[np.identity(2), np.identity(2)], [unitary_mat, -unitary_mat]]

    np.testing.assert_equal(is_completely_positive(kraus_ops), False)


def test_is_completely_positive_choi_true():
    """Verify Choi matrix of the depolarizing map is completely positive."""
    np.testing.assert_equal(is_completely_positive(depolarizing(2)), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
