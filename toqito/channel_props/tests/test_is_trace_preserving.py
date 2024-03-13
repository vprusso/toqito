"""Tests for is_trace_preserving."""

import numpy as np
import pytest

from toqito.channel_props import is_trace_preserving
from toqito.channels import depolarizing

unitary_mat = np.array([[1, 1], [-1, 1]]) / np.sqrt(2)
kraus_ops = [[np.identity(2), np.identity(2)], [unitary_mat, -unitary_mat]]


@pytest.mark.parametrize(
    "input_unitary, expected_result, dims",
    [(kraus_ops, False, None), (depolarizing(2), True, None), (depolarizing(4), True, 4), (depolarizing(4), False, 2)],
)
def test_is_trace_prserving(input_unitary, expected_result, dims):
    """Test function works as expected."""
    assert is_trace_preserving(input_unitary, dim=dims) == expected_result
