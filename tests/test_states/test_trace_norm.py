"""Tests for trace_norm function."""
import unittest
import numpy as np
from toqito.states import basis
from toqito.state_metrics import trace_norm


class TestTraceNorm(unittest.TestCase):

    """Units test for trace_norm."""

    def test_trace_norm(self):
        """Test trace norm."""
        e_0, e_1 = basis(2, 0), basis(2, 1)
        e_00 = np.kron(e_0, e_0)
        e_11 = np.kron(e_1, e_1)

        u_vec = 1 / np.sqrt(2) * (e_00 + e_11)
        rho = u_vec * u_vec.conj().T

        res = trace_norm(rho)
        _, singular_vals, _ = np.linalg.svd(rho)
        expected_res = float(np.sum(singular_vals))

        self.assertEqual(np.isclose(res, expected_res), True)


if __name__ == "__main__":
    unittest.main()
