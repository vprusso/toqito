"""Tests for trace_distance function."""
import unittest
import numpy as np
from toqito.base.ket import ket
from toqito.state.distance.trace_distance import trace_distance


class TestTraceDistance(unittest.TestCase):
    """Unit test for trace_distance."""

    def test_trace_distance_same_state(self):
        r"""Test that: :math: `T(\rho, \sigma) = 0` iff `\rho = \sigma`."""
        e_0, e_1 = ket(2, 0), ket(2, 1)
        e_00 = np.kron(e_0, e_0)
        e_11 = np.kron(e_1, e_1)

        u_vec = 1 / np.sqrt(2) * (e_00 + e_11)
        rho = u_vec * u_vec.conj().T
        sigma = rho

        res = trace_distance(rho, sigma)

        self.assertEqual(np.isclose(res, 0), True)


if __name__ == '__main__':
    unittest.main()
