"""Tests for bell_inequality_max."""

import numpy as np
import pytest

from toqito.state_opt import bell_inequality_max


@pytest.mark.parametrize(
    "joint_coe, a_coe, b_coe, a_val, b_val, expected",
    # Bell I3322 inequality.
    [
        (
            np.array([[1, 1, -1], [1, 1, 1], [-1, 1, 0]]),
            np.array([0, -1, 0]),
            np.array([-1, -2, 0]),
            np.array([0, 1]),
            np.array([0, 1]),
            0.250,
        ),
        # Bell CHSH inequality.
        (
            np.array([[1, 1], [1, -1]]),
            np.array([0, 0]),
            np.array([0, 0]),
            np.array([1, -1]),
            np.array([1, -1]),
            2 * np.sqrt(2),
        ),
    ],
)
def test_bell_inequality_max_valid(joint_coe, a_coe, b_coe, a_val, b_val, expected):
    """Test bell_inequality_max returns the expected value using valid input."""
    result = bell_inequality_max(joint_coe, a_coe, b_coe, a_val, b_val)
    assert pytest.approx(result, 0.01) == expected


@pytest.mark.parametrize(
    "joint_coe, a_coe, b_coe, a_val, b_val",
    [
        (
            np.array([[1, 1, -1], [1, 1, 1], [-1, 1, 0]]),
            np.array([0, -1, 0]),
            np.array([-1, -2, 0]),
            np.array([0, 1, 0]),
            np.array([0, 1, 0]),
        )
    ],
)
def test_bell_inequality_max_invalid(joint_coe, a_coe, b_coe, a_val, b_val):
    """Test bell_inequality_max raises ValueError when the measurement outcome arrays do not have length 2.

    This test ensures that the function correctly raises the ValueError.
    """
    with pytest.raises(ValueError):
        bell_inequality_max(joint_coe, a_coe, b_coe, a_val, b_val)


@pytest.mark.parametrize(
    "joint_coe, a_coe, b_coe, a_val, b_val, expected",
    [(np.zeros((1, 1)), np.zeros(1), np.zeros(1), np.array([0, 1]), np.array([0, 1]), 0.0)],
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
