"""Tests for completely_bounded_trace_norm."""

import numpy as np
import pytest

from toqito.channel_metrics import completely_bounded_trace_norm
from toqito.channel_ops import kraus_to_choi
from toqito.channels.dephasing import dephasing

# things required for cb trace norm in terms of the eigenvalues of U
U = 1 / np.sqrt(2) * np.array([[1, 1], [-1, 1]])  # Hadamard gate
phi = kraus_to_choi([[np.eye(2), np.eye(2)], [U, -U]])
lam, _ = np.linalg.eig(U)
dist = np.abs(lam[:, None] - lam[None, :])  # all to all distance
diameter = np.max(dist)

# A qutrit analogue: the difference of the clock-unitary channel and the identity channel. The
# convex hull of the eigenvalues {1, w, w^2} of the clock unitary contains the origin, so the
# completely bounded trace norm of the difference is 2.
omega = np.exp(2j * np.pi / 3)
clock = np.diag([1, omega, omega**2])
phi_qutrit = kraus_to_choi([[np.eye(3), np.eye(3)], [clock, -clock]])

# Isometries from a qubit into a qutrit, used to build channels with unequal input and output
# dimensions.
isometry_1 = np.array([[1, 0], [0, 1], [0, 0]])
isometry_2 = np.array([[1, 0], [0, 0], [0, 1]])

solvers = ["cvxopt"]


@pytest.mark.parametrize(
    "test_input, expected",
    [
        # The diamond norm of a quantum channel is 1
        (dephasing(2), 1),
        # the diamond norm of a CP map
        (np.eye(4), 4.0),
        # unitaries channel, phi, diameter in terms of the eigenvalues of U
        (phi, diameter),
        # qutrit unitaries channel (non-power-of-2 dimension)
        (phi_qutrit, 2),
    ],
)
@pytest.mark.parametrize("solver", solvers)
def test_cb_trace_norm(test_input, solver, expected):
    """Test function works as expected for valid inputs."""
    calculated_value = completely_bounded_trace_norm(test_input, solver)
    assert abs(calculated_value - expected) <= 1e-3


@pytest.mark.parametrize(
    "test_input, expected",
    [
        # A channel with unequal input and output dimensions (a qubit-to-qutrit isometry
        # channel) is a quantum channel, so its completely bounded trace norm is 1.
        (kraus_to_choi([[isometry_1, isometry_1]]), 1),
        # The difference of two isometry channels whose isometries map |1> to orthogonal states
        # is perfectly distinguishable, so its completely bounded trace norm is 2.
        (kraus_to_choi([[isometry_1, isometry_1]]) - kraus_to_choi([[isometry_2, isometry_2]]), 2),
    ],
)
def test_cb_trace_norm_rectangular(test_input, expected):
    """Channels with unequal input/output dimensions are handled via `dim` (issue #1596)."""
    calculated_value = completely_bounded_trace_norm(test_input, dim=[2, 3])
    assert abs(calculated_value - expected) <= 1e-3


def test_cb_trace_norm_rectangular_requires_dim():
    """A Choi matrix with unequal input/output dimensions requires the `dim` argument."""
    choi = kraus_to_choi([[isometry_1, isometry_1]])
    with pytest.raises(ValueError, match="the optional argument DIM must be specified"):
        completely_bounded_trace_norm(choi)


def test_cb_trace_norm_invalid_input():
    """Non-square inputs for cb trace norm."""
    with pytest.raises(
        ValueError,
        match="The input and output spaces of the superoperator phi must both be square.",
    ):
        phi1 = np.array([[1, 2, 3], [4, 5, 6]])
        completely_bounded_trace_norm(phi1)
