"""Tests for bell_inequality_max."""

import numpy as np
import pytest
import scs

from toqito.state_opt import bell_inequality_max

# Example inputs from the I3322 Bell inequality
joint_coe = np.array([[1, 1, -1], [1, 1, 1], [-1, 1, 0]])
a_coe = np.array([0, -1, 0])
b_coe = np.array([-1, -2, 0])
a_val_valid = np.array([0, 1])
b_val_valid = np.array([0, 1])
expected_value = 0.250


@pytest.mark.parametrize(
    "joint_coe, a_coe, b_coe, a_val, b_val, expected",
    [(joint_coe, a_coe, b_coe, a_val_valid, b_val_valid, expected_value)],
)
def test_bell_inequality_max_valid(joint_coe, a_coe, b_coe, a_val, b_val, expected):
    """Test bell_inequality_max returns the expected value using valid input.

    This test uses the I3322 Bell inequality.
    """
    result = bell_inequality_max(joint_coe, a_coe, b_coe, a_val, b_val)
    assert pytest.approx(result, 0.01) == expected


@pytest.mark.parametrize(
    "joint_coe, a_coe, b_coe, a_val, b_val",
    [
        # a_val has invalid length (3 instead of 2)
        (joint_coe, a_coe, b_coe, np.array([0, 1, 2]), b_val_valid),
        # b_val has invalid length (3 instead of 2)
        (joint_coe, a_coe, b_coe, a_val_valid, np.array([0, 1, 2])),
    ],
)
def test_bell_inequality_max_invalid(joint_coe, a_coe, b_coe, a_val, b_val):
    """Test bell_inequality_max raises ValueError when the measurement outcome arrays do not have length 2.

    This test ensures that the function correctly raises the ValueError.
    """
    with pytest.raises(ValueError):
        bell_inequality_max(joint_coe, a_coe, b_coe, a_val, b_val)


# Degenerate case: all coefficients are zero (using minimal m = 1)
degenerate_joint_coe = np.zeros((1, 1))
degenerate_a_coe = np.zeros(1)
degenerate_b_coe = np.zeros(1)


@pytest.mark.parametrize(
    "joint_coe, a_coe, b_coe, a_val, b_val, expected",
    [(degenerate_joint_coe, degenerate_a_coe, degenerate_b_coe, np.array([0, 1]), np.array([0, 1]), 0.0)],
)
def test_bell_inequality_max_degenerate(joint_coe, a_coe, b_coe, a_val, b_val, expected):
    """Test bell_inequality_max for the degenerate case where all coefficient values are zero.

    In this case, the objective matrix is zero and the SDP should yield 0.
    """
    result = bell_inequality_max(joint_coe, a_coe, b_coe, a_val, b_val)
    assert pytest.approx(result, 0.01) == expected


# Edge case: minimal measurement settings (m = 1) with nonzero joint coefficient.
minimal_joint_coe = np.array([[1]])
minimal_a_coe = np.array([0])
minimal_b_coe = np.array([0])


@pytest.mark.parametrize(
    "joint_coe, a_coe, b_coe, a_val, b_val",
    [(minimal_joint_coe, minimal_a_coe, minimal_b_coe, np.array([0, 1]), np.array([0, 1]))],
)
def test_bell_inequality_max_minimal(joint_coe, a_coe, b_coe, a_val, b_val):
    """Test bell_inequality_max on a minimal measurement settings (m = 1) case with nonzero joint coefficient.

    Since the exact optimal value is non-trivial to calculate by hand,
    this test ensures the function returns a finite, non-negative float.
    """
    result = bell_inequality_max(joint_coe, a_coe, b_coe, a_val, b_val)
    assert isinstance(result, float)
    assert result >= 0


# CHSH inequality test


@pytest.mark.parametrize(
    "joint_coe, a_coe, b_coe, a_val, b_val, expected",
    [(chsh_joint_coe, chsh_a_coe, chsh_b_coe, chsh_a_val, chsh_b_val, chsh_expected)],
)
def test_bell_inequality_max_chsh(joint_coe, a_coe, b_coe, a_val, b_val, expected):
    """Test bell_inequality_max with CHSH inequality.

    The quantum mechanical upper bound for CHSH is Tsirelson’s bound ≈ 2.828.
    """
    result = bell_inequality_max(joint_coe, a_coe, b_coe, a_val, b_val)
    assert pytest.approx(result, 0.01) == expected
