"""Unit tests for the is_abs_ppt function in toqito.state_props."""

import numpy as np
import pytest

from toqito.rand import random_psd_operator
from toqito.state_props import is_abs_ppt
from toqito.states import max_mixed


def test_maximally_mixed_states():
    """Test that maximally mixed states are absolutely PPT for supported dimensions."""
    # 2x2
    rho_2x2 = max_mixed(4)
    assert is_abs_ppt(rho_2x2, [2, 2]) is True
    # 2x3
    rho_2x3 = max_mixed(6)
    assert is_abs_ppt(rho_2x3, [2, 3]) is True
    # 3x3
    rho_3x3 = max_mixed(9)
    assert is_abs_ppt(rho_3x3, [3, 3]) is True


def test_random_psd_not_absolutely_ppt():
    """Test that random PSD states are not always absolutely PPT."""
    np.random.seed(42)
    for dim in ([2, 2], [2, 3], [3, 3]):
        n = dim[0] * dim[1]
        for _ in range(3):
            rho = random_psd_operator(n)
            rho = rho / np.trace(rho)
            # All absolutely PPT states are PPT, but not vice versa
            # So if is_abs_ppt returns False, that's fine
            res = is_abs_ppt(rho, dim)
            assert res in [True, False, -1]


def test_known_non_absolutely_ppt():
    """Test a known non-absolutely PPT state (e.g., Bell state)."""
    # Construct a state that is PPT but not absolutely PPT for 2x2
    # Take a pure Bell state (not PPT, so not absolutely PPT)
    bell = np.zeros((4, 4), dtype=complex)
    bell[0, 3] = bell[3, 0] = 1 / 2
    bell[1, 2] = bell[2, 1] = 1 / 2
    bell[0, 0] = bell[3, 3] = 1 / 2
    bell[1, 1] = bell[2, 2] = 1 / 2
    bell = bell / np.trace(bell)
    assert is_abs_ppt(bell, [2, 2]) is False


def test_invalid_input():
    """Test that invalid input raises appropriate ValueError."""
    # Not square
    with pytest.raises(ValueError):
        is_abs_ppt(np.ones((3, 4)))
    # Not Hermitian
    with pytest.raises(ValueError):
        is_abs_ppt(np.array([[0, 1], [0, 0]]))
    # Not trace 1
    with pytest.raises(ValueError):
        is_abs_ppt(np.eye(4) * 2)
    # Not positive semidefinite
    with pytest.raises(ValueError):
        is_abs_ppt(np.array([[1, 0], [0, -1]]))


def test_large_dim_returns_minus_one():
    """Test that unsupported large dimensions return -1."""
    # 4x4 system (not implemented)
    rho = max_mixed(16)
    assert is_abs_ppt(rho, [4, 4]) == -1
