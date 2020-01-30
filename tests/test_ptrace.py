import unittest
import numpy as np

from toqito.super_operators.ptrace import ptrace
from toqito.helper.constants import e01, e10


class TestPtrace(unittest.TestCase):
    """Unit test for ptrace."""

    def test_ptrace_1(self):
        """Test for ptrace."""
        psi = 1/np.sqrt(2)*(e01 - e10)
        rho = psi * psi.conj().T

        res = ptrace(rho, dims=[2, 2], axis=0)
        expected_res = np.array([[1/2, 0], [0, 1/2]])

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)

    def test_ptrace_2(self):
        # Generate the results we want
        rho_A = np.random.rand(4, 4) + 1j*np.random.rand(4, 4)
        rho_A /= np.trace(rho_A)
        rho_B = np.random.rand(3, 3) + 1j*np.random.rand(3, 3)
        rho_B /= np.trace(rho_B)
        rho_C = np.random.rand(2, 2) + 1j*np.random.rand(2, 2)
        rho_C /= np.trace(rho_C)

        rho_AB = np.kron(rho_A, rho_B)
        rho_AC = np.kron(rho_A, rho_C)
        rho_ABC = np.kron(rho_AB, rho_C)

        # Try to get the results by doing partial_trace's from rho_ABC
        rho_AB_test = ptrace(rho_ABC, [4, 3, 2], axis=2)
        rho_AC_test = ptrace(rho_ABC, [4, 3, 2], axis=1)
        rho_A_test = ptrace(rho_AB_test, [4, 3], axis=1)
        rho_B_test = ptrace(rho_AB_test, [4, 3], axis=0)
        rho_C_test = ptrace(rho_AC_test, [4, 2], axis=0)

        # See if the outputs of partial_trace are correct
        assert np.allclose(rho_AB_test, rho_AB)
        assert np.allclose(rho_AC_test, rho_AC)
        assert np.allclose(rho_A_test, rho_A)
        assert np.allclose(rho_B_test, rho_B)
        assert np.allclose(rho_C_test, rho_C)


if __name__ == '__main__':
    unittest.main()
