"""Unit tests for the is_abs_ppt function in toqito.state_props."""

import numpy as np
import pytest

from toqito.rand import random_psd_operator
from toqito.state_props import is_abs_ppt
from toqito.states import max_mixed


@pytest.mark.parametrize(
    "dims, n",
    [([2, 2], 4), ([2, 3], 6), ([3, 3], 9)]
)
def test_maximally_mixed_states(dims, n):
    """Test that maximally mixed states are absolutely PPT for supported dimensions."""
    rho = max_mixed(n)
    assert is_abs_ppt(rho, dims) is True


@pytest.mark.parametrize("dims", [([2, 2]), ([2, 3]), ([3, 3])])
def test_random_psd_not_absolutely_ppt(dims):
    """Test that random PSD states are not always absolutely PPT."""
    np.random.seed(42)
    n = dims[0] * dims[1]
    for _ in range(3):
        rho = random_psd_operator(n)
        rho = rho / np.trace(rho)
        res = is_abs_ppt(rho, dims)
        assert res in [True, False, -1]


def test_known_non_absolutely_ppt():
    """Test a known non-absolutely PPT state (e.g., Bell state)."""
    bell = np.zeros((4, 4), dtype=complex)
    bell[0, 3] = bell[3, 0] = 1 / 2
    bell[1, 2] = bell[2, 1] = 1 / 2
    bell[0, 0] = bell[3, 3] = 1 / 2
    bell[1, 1] = bell[2, 2] = 1 / 2
    bell = bell / np.trace(bell)
    assert is_abs_ppt(bell, [2, 2]) is False


import re

@pytest.mark.parametrize(
    "matrix, error_msg",
    [
        (np.ones((3, 4)), "Input matrix must be square."),
        (np.array([[0, 1], [0, 0]]), "Input matrix must be Hermitian."),
        (np.eye(4) * 2, "Input matrix must have trace 1"),
        (np.array([[1, 0], [0, -1]]), "Input matrix must be positive semidefinite."),
    ]
)
def test_invalid_input(matrix, error_msg):
    """Test that invalid input raises appropriate ValueError with correct message."""
    with pytest.raises(ValueError, match=re.escape(error_msg)):
        is_abs_ppt(matrix)


def test_large_dim_returns_minus_one():
    """Test that unsupported large dimensions return -1."""
    rho = max_mixed(16)
    assert is_abs_ppt(rho, [4, 4]) == -1
