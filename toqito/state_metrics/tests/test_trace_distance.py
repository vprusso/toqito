"""Tests for trace_distance."""

import numpy as np

from toqito.state_metrics import trace_distance
from toqito.states import basis


def test_trace_distance_same_state():
    r"""Test that: :math:`T(\rho, \sigma) = 0` iff `\rho = \sigma`."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    e_00 = np.kron(e_0, e_0)
    e_11 = np.kron(e_1, e_1)

    u_vec = 1 / np.sqrt(2) * (e_00 + e_11)
    rho = u_vec @ u_vec.conj().T
    sigma = rho

    res = trace_distance(rho, sigma)

    np.testing.assert_equal(np.isclose(res, 0), True)


def test_trace_distance_orthogonal_pure_states():
    r"""Trace distance between orthogonal pure states is 1."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    rho = e_0 @ e_0.conj().T
    sigma = e_1 @ e_1.conj().T

    np.testing.assert_allclose(trace_distance(rho, sigma), 1)


def test_trace_distance_non_commuting_states():
    r"""Trace distance equals half the sum of \|eig(rho - sigma)\| for non-commuting states."""
    rho = np.array([[0.6, 0.2], [0.2, 0.4]])
    sigma = np.array([[0.4, 0.3], [0.3, 0.6]])

    res = trace_distance(rho, sigma)

    # The elementwise-abs implementation returned 0.2 here; the correct value is ~0.2236.
    expected = 0.5 * np.sum(np.abs(np.linalg.eigvalsh(rho - sigma)))
    np.testing.assert_allclose(res, expected)


def test_trace_distance_non_density_matrix():
    r"""Test trace distance on non-density matrix."""
    rho = np.array([[1, 2], [3, 4]])
    sigma = np.array([[5, 6], [7, 8]])

    with np.testing.assert_raises(ValueError):
        trace_distance(rho, sigma)
