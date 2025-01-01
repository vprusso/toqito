"""Tests for trace_distance."""

import numpy as np

from toqito.matrices import standard_basis
from toqito.state_metrics import trace_distance


def test_trace_distance_same_state():
    r"""Test that: :math:`T(\rho, \sigma) = 0` iff `\rho = \sigma`."""
    e_0, e_1 = standard_basis(2)
    e_00 = np.kron(e_0, e_0)
    e_11 = np.kron(e_1, e_1)

    u_vec = 1 / np.sqrt(2) * (e_00 + e_11)
    rho = u_vec @ u_vec.conj().T
    sigma = rho

    res = trace_distance(rho, sigma)

    np.testing.assert_equal(np.isclose(res, 0), True)


def test_trace_distance_non_density_matrix():
    r"""Test trace distance on non-density matrix."""
    rho = np.array([[1, 2], [3, 4]])
    sigma = np.array([[5, 6], [7, 8]])

    with np.testing.assert_raises(ValueError):
        trace_distance(rho, sigma)
