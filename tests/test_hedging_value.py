"""Tests for hedging_value function."""
import unittest
from numpy import kron, cos, sin, pi, sqrt, isclose

from toqito.core.ket import ket
from toqito.nonlocal_games.quantum_hedging.hedging_value import HedgingValue


class TestHedgingValue(unittest.TestCase):
    """Unit test for hedging_value."""

    e_0, e_1 = ket(2, 0), ket(2, 1)
    e_00, e_01 = kron(e_0, e_0), kron(e_0, e_1)
    e_10, e_11 = kron(e_1, e_0), kron(e_1, e_1)

    alpha = 1 / sqrt(2)
    theta = pi / 8

    w_var = alpha * cos(theta) * e_00 + sqrt(1 - alpha ** 2) * sin(theta) * e_11

    l_1 = -alpha * sin(theta) * e_00 + sqrt(1 - alpha ** 2) * cos(theta) * e_11

    l_2 = alpha * sin(theta) * e_10

    l_3 = sqrt(1 - alpha ** 2) * cos(theta) * e_01

    q_1 = w_var * w_var.conj().T
    q_0 = l_1 * l_1.conj().T + l_2 * l_2.conj().T + l_3 * l_3.conj().T

    def test_max_prob_outcome_a_primal_1_dim(self):
        """
        Maximal probability of outcome "a" when dim == 1.

        The primal problem of the hedging semidefinite program.
        """
        q_0 = TestHedgingValue.q_0
        hedging_value = HedgingValue(q_0, 1)
        self.assertEqual(
            isclose(hedging_value.max_prob_outcome_a_primal(), cos(pi / 8) ** 2), True
        )

    def test_max_prob_outcome_a_primal_2_dim(self):
        """
        Test maximal probability of outcome "a" when dim == 2.

        The primal problem of the hedging semidefinite program.
        """
        q_00 = kron(TestHedgingValue.q_0, TestHedgingValue.q_0)
        hedging_value = HedgingValue(q_00, 2)
        self.assertEqual(
            isclose(hedging_value.max_prob_outcome_a_primal(), cos(pi / 8) ** 4), True
        )

    def test_max_prob_outcome_a_dual_1_dim(self):
        """
        Test maximal probability of outcome "a" when dim == 1.

        The dual problem of the hedging semidefinite program.
        """
        q_0 = TestHedgingValue.q_0
        hedging_value = HedgingValue(q_0, 1)
        self.assertEqual(
            isclose(hedging_value.max_prob_outcome_a_dual(), cos(pi / 8) ** 2), True
        )

    def test_max_prob_outcome_a_dual_2_dim(self):
        """
        Test maximal probability of outcome "a" when dim == 2.

        The dual problem of the hedging semidefinite program.
        """
        q_00 = kron(TestHedgingValue.q_0, TestHedgingValue.q_0)
        hedging_value = HedgingValue(q_00, 2)
        self.assertEqual(
            isclose(hedging_value.max_prob_outcome_a_dual(), cos(pi / 8) ** 4), True
        )

    def test_min_prob_outcome_a_primal_1_dim(self):
        """
        Test minimal probability of outcome "a" when dim == 1.

        The primal problem of the hedging semidefinite program.
        """
        q_1 = TestHedgingValue.q_1
        hedging_value = HedgingValue(q_1, 1)
        self.assertEqual(
            isclose(hedging_value.min_prob_outcome_a_primal(), 0, atol=0.01), True
        )

    def test_min_prob_outcome_a_primal_2_dim(self):
        """
        Test minimal probability of outcome "a" when dim == 2.

        The primal problem of the hedging semidefinite program.
        """
        q_11 = kron(TestHedgingValue.q_1, TestHedgingValue.q_1)
        hedging_value = HedgingValue(q_11, 2)
        self.assertEqual(
            isclose(hedging_value.min_prob_outcome_a_primal(), 0, atol=0.01), True
        )

    def test_min_prob_outcome_a_dual_1_dim(self):
        """
        Test minimal probability of outcome "a" when dim == 1.

        The dual problem of the hedging semidefinite program.
        """
        q_1 = TestHedgingValue.q_1
        hedging_value = HedgingValue(q_1, 1)
        self.assertEqual(
            isclose(hedging_value.min_prob_outcome_a_dual(), 0, atol=0.01), True
        )

    def test_min_prob_outcome_a_dual_2_dim(self):
        """
        Test minimal probability of outcome "a" when dim == 2.

        The dual problem of the hedging semidefinite program.
        """
        q_11 = kron(TestHedgingValue.q_1, TestHedgingValue.q_1)
        hedging_value = HedgingValue(q_11, 2)
        self.assertEqual(
            isclose(hedging_value.min_prob_outcome_a_dual(), 0, atol=0.01), True
        )


if __name__ == "__main__":
    unittest.main()
