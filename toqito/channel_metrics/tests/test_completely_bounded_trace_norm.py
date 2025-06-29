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
    ],
)
@pytest.mark.parametrize("solver", solvers)
def test_cb_trace_norm(test_input, solver, expected):
    """Test function works as expected for valid inputs."""
    calculated_value = completely_bounded_trace_norm(test_input, solver)
    assert abs(calculated_value - expected) <= 1e-3


def test_cb_trace_norm_invalid_input():
    """Non-square inputs for cb trace norm."""
    with pytest.raises(
        ValueError,
        match="The input and output spaces of the superoperator phi must both be square.",
    ):
        phi1 = np.array([[1, 2, 3], [4, 5, 6]])
        completely_bounded_trace_norm(phi1)
