"""Tests for is_trace_preserving."""
import numpy as np

from toqito.channel_props import is_trace_preserving
from toqito.channels import depolarizing


def test_is_trace_preserving_kraus_false():
    """Verify non-trace preserving channel as Kraus ops as False."""
    unitary_mat = np.array([[1, 1], [-1, 1]]) / np.sqrt(2)
    kraus_ops = [[np.identity(2), np.identity(2)], [unitary_mat, -unitary_mat]]

    np.testing.assert_equal(is_trace_preserving(kraus_ops), False)


def test_is_trace_preserving_choi_true():
    """Verify Choi matrix of the depolarizing map is trace preserving."""
    np.testing.assert_equal(is_trace_preserving(depolarizing(2)), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
