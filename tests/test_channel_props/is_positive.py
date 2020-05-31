"""Tests for is_positive."""
import numpy as np

from toqito.channel_props import is_positive
from toqito.channels import depolarizing


def test_is_positive_kraus_false():
    """Verify non-completely positive channel as Kraus ops as False."""
    unitary_mat = np.array([[1, 1], [-1, -1]]) / np.sqrt(2)
    kraus_ops = [[np.identity(2), np.identity(2)], [unitary_mat, -unitary_mat]]

    np.testing.assert_equal(is_positive(kraus_ops), False)


def test_is_positive_choi_true():
    """Verify that the Choi matrix of the depolarizing map is positive."""
    np.testing.assert_equal(is_positive(depolarizing(4)), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
