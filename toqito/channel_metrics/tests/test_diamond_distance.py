"""Tests for diamond_norm."""

import numpy as np
import pytest

from toqito.channel_metrics import diamond_distance
from toqito.channel_ops import kraus_to_choi
from toqito.channels import dephasing, depolarizing


@pytest.mark.parametrize(
    "test_input1, test_input_2, expected",
    [
        # The diamond norm of identical channels should yield 0
        (dephasing(2), dephasing(2), 0),
        # the diamond norm of different channels
        (dephasing(2), depolarizing(2), 1),
    ],
)
def test_diamond_norm_valid_inputs(test_input1, test_input_2, expected):
    """Test function works as expected for valid inputs."""
    calculated_value = diamond_distance(test_input1, test_input_2)
    assert pytest.approx(expected, 1e-3) == calculated_value


def test_diamond_norm_qutrit_unitaries():
    """Qutrit unitary channels (a non-power-of-2 dimension) are handled correctly (issue #1596).

    For unitary channels the diamond distance is determined by the eigenvalues of U^dagger V.
    For U = diag(1, 1, exp(i * theta)) versus the identity, it equals 2 * sin(theta / 2).
    """
    theta = np.pi / 2
    unitary = np.diag([1, 1, np.exp(1j * theta)])
    choi_u = kraus_to_choi([[unitary, unitary]])
    choi_id = kraus_to_choi([[np.eye(3), np.eye(3)]])
    assert pytest.approx(2 * np.sin(theta / 2), abs=1e-3) == diamond_distance(choi_u, choi_id)


@pytest.mark.parametrize(
    "test_input_2, expected",
    [
        # The two isometries map |1> to orthogonal states, so the channels are perfectly
        # distinguishable and the diamond distance is 2.
        (np.array([[1, 0], [0, 0], [0, 1]]), 2),
        # V2 = |0><0| + (cos(t)|1> + sin(t)|2>)<1| with t = pi / 6. For isometry channels the
        # diamond distance is 2 * sin(t) = 1.
        (np.array([[1, 0], [0, np.cos(np.pi / 6)], [0, np.sin(np.pi / 6)]]), 1),
    ],
)
def test_diamond_norm_rectangular_channels(test_input_2, expected):
    """Channels with unequal input/output dimensions are handled via `dim` (issue #1596)."""
    isometry_1 = np.array([[1, 0], [0, 1], [0, 0]])
    choi_1 = kraus_to_choi([[isometry_1, isometry_1]])
    choi_2 = kraus_to_choi([[test_input_2, test_input_2]])
    assert pytest.approx(expected, abs=1e-3) == diamond_distance(choi_1, choi_2, dim=[2, 3])


@pytest.mark.parametrize(
    "test_input1, test_input_2, expected_msg",
    [
        # Inconsistent dimensions between Choi matrices
        (depolarizing(4), dephasing(2), r"operands could not be broadcast together with shapes \(16,16\) \(4,4\)"),
        # Non-square inputs for diamond norm
        (
            np.array([[1, 2, 3], [4, 5, 6]]),
            np.array([[1, 2, 3], [4, 5, 6]]),
            "The input and output spaces of the superoperator phi must both be square.",
        ),
    ],
)
def test_diamond_norm_invalid_inputs(test_input1, test_input_2, expected_msg):
    """Test function raises error as expected for invalid inputs."""
    with pytest.raises(ValueError, match=expected_msg):
        diamond_distance(test_input1, test_input_2)
