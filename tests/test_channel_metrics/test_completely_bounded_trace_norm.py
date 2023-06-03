"""Tests for completely_bounded_trace_norm"""
import numpy as np

from toqito.channels.dephasing import dephasing
from toqito.channel_metrics.completely_bounded_trace_norm import completely_bounded_trace_norm


def test_cb_trace_norm_quantum_channel():
    """The diamond norm of a quantum channel is 1"""
    phi = dephasing(2)
    np.testing.assert_equal(completely_bounded_trace_norm(phi), 1)


def test_cb_trace_norm_unitaries_channel():
    """The diamond norm of phi = id- U id U* is the diameter of the smallest circle that contains the eigenvalues of U"""
    U = 1/np.sqrt(2) * np.array([[1, 1], [-1, 1]]) # Hadamard gate
    phi = np.block([[np.eye(2), np.eye(2)],[U, -U]])
    lam, eigv = np.linalg.eig(U)
    dist = np.abs(lam[:, None] - lam[None, :]) # all to all distance
    diameter =  np.max(dist)
    np.testing.assert_equal(np.isclose(completely_bounded_trace_norm(phi), diameter, atol=1e-3), True)


def test_cb_trace_norm_non_square():
    """Non-square inputs for cb trace norm."""
    with np.testing.assert_raises(ValueError):
        phi = np.array([[1, 2, 3], [4, 5, 6]])
        completely_bounded_trace_norm(phi)


if __name__ == "__main__":
    np.testing.run_module_suite()