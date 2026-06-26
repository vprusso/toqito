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


def test_is_trace_preserving_invalid_dim_raises():
    """Ensure that a ValueError is raised when the input matrix has non-square dimension."""
    with pytest.raises(ValueError, match="Cannot infer equal subsystem dimensions. Please provide `dim`."):
        is_trace_preserving(np.eye(3))


def test_is_trace_preserving_flat_kraus_format():
    """The flat Kraus format [K1, K2, ...] is accepted (sum K_i^dagger K_i = I)."""
    kraus = [np.eye(2) / np.sqrt(2), np.array([[0, 1], [1, 0]]) / np.sqrt(2)]
    assert is_trace_preserving(kraus) is True
    assert is_trace_preserving([0.5 * np.eye(2)]) is False
