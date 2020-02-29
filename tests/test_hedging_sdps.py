"""Tests for hedging_sdps function."""
import unittest
from numpy import kron, cos, sin, pi, sqrt, isclose

from toqito.base.ket import ket
from toqito.nonlocal_games.hedging.hedging_sdps import max_prob_outcome_a_primal
from toqito.nonlocal_games .hedging.hedging_sdps import max_prob_outcome_a_dual
from toqito.nonlocal_games.hedging.hedging_sdps import min_prob_outcome_a_primal
from toqito.nonlocal_games.hedging.hedging_sdps import min_prob_outcome_a_dual


class TestHedgingSDPs(unittest.TestCase):
    """Unit test for hedging_sdps."""
    e_0, e_1 = ket(2, 0), ket(2, 1)
    e_00, e_01 = kron(e_0, e_0), kron(e_0, e_1)
    e_10, e_11 = kron(e_1, e_0), kron(e_1, e_1)

    alpha = 1/sqrt(2)
    theta = pi/8

    w_var = alpha * cos(theta) * e_00 + \
        sqrt(1 - alpha**2) * sin(theta) * e_11

    l_1 = -alpha * sin(theta) * e_00 + \
        sqrt(1 - alpha**2) * cos(theta) * e_11

    l_2 = alpha * sin(theta) * e_10

    l_3 = sqrt(1 - alpha**2) * cos(theta) * e_01

    q_1 = w_var * w_var.conj().T
    q_0 = l_1 * l_1.conj().T + \
        l_2 * l_2.conj().T + \
        l_3 * l_3.conj().T

    def test_max_prob_outcome_a_primal_1_dim(self):
        """
        Test maximal probability of outcome "a" when dim == 1 using the primal
        problem of the hedging SDP.
        """
        q_0 = TestHedgingSDPs.q_0
        self.assertEqual(isclose(max_prob_outcome_a_primal(q_0, 1),
                                 cos(pi/8)**2), True)

    def test_max_prob_outcome_a_primal_2_dim(self):
        """
        Test maximal probability of outcome "a" when dim == 2 using the primal
        problem of the hedging SDP.
        """
        q_00 = kron(TestHedgingSDPs.q_0, TestHedgingSDPs.q_0)
        self.assertEqual(isclose(max_prob_outcome_a_primal(q_00, 2),
                                 cos(pi/8)**4), True)

    def test_max_prob_outcome_a_dual_1_dim(self):
        """
        Test maximal probability of outcome "a" when dim == 1 using the dual
        problem of the hedging SDP.
        """
        q_0 = TestHedgingSDPs.q_0
        self.assertEqual(isclose(max_prob_outcome_a_dual(q_0, 1),
                                 cos(pi/8)**2), True)

    def test_max_prob_outcome_a_dual_2_dim(self):
        """
        Test maximal probability of outcome "a" when dim == 2 using the dual
        problem of the hedging SDP.
        """
        q_00 = kron(TestHedgingSDPs.q_0, TestHedgingSDPs.q_0)
        self.assertEqual(isclose(max_prob_outcome_a_dual(q_00, 2),
                                 cos(pi/8)**4), True)

    def test_min_prob_outcome_a_primal_1_dim(self):
        """
        Test minimal probability of outcome "a" when dim == 1 using the primal
        problem of the hedging SDP.
        """
        q_1 = TestHedgingSDPs.q_1
        self.assertEqual(isclose(min_prob_outcome_a_primal(q_1, 1),
                                 0, atol=0.01), True)

    def test_min_prob_outcome_a_primal_2_dim(self):
        """
        Test minimal probability of outcome "a" when dim == 2 using the primal
        problem of the hedging SDP.
        """
        q_11 = kron(TestHedgingSDPs.q_1, TestHedgingSDPs.q_1)
        self.assertEqual(isclose(min_prob_outcome_a_primal(q_11, 2),
                                 0, atol=0.01), True)

    def test_min_prob_outcome_a_dual_1_dim(self):
        """
        Test minimal probability of outcome "a" when dim == 1 using the dual
        problem of the hedging SDP.
        """
        q_1 = TestHedgingSDPs.q_1
        self.assertEqual(isclose(min_prob_outcome_a_dual(q_1, 1),
                                 0, atol=0.01), True)

    def test_min_prob_outcome_a_dual_2_dim(self):
        """
        Test minimal probability of outcome "a" when dim == 2 using the dual
        problem of the hedging SDP.
        """
        q_11 = kron(TestHedgingSDPs.q_1, TestHedgingSDPs.q_1)
        self.assertEqual(isclose(min_prob_outcome_a_dual(q_11, 2),
                                 0, atol=0.01), True)


if __name__ == '__main__':
    unittest.main()
