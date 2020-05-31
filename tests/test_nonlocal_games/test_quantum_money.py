"""Tests for QuantumMoney class."""
import unittest
import pytest
import numpy as np

from toqito.states import basis
from toqito.nonlocal_games.quantum_money import QuantumMoney


class TestQuantumMoney(unittest.TestCase):
    """Unit test for counterfeit_attack."""

    e_0, e_1 = basis(2, 0), basis(2, 1)
    e_p = (e_0 + e_1) / np.sqrt(2)
    e_m = (e_0 - e_1) / np.sqrt(2)

    def test_counterfeit_attack_wiesner_money(self):
        """Probability of counterfeit attack on Wiesner's quantum money."""
        states = [self.e_0, self.e_1, self.e_p, self.e_m]
        probs = [1 / 4, 1 / 4, 1 / 4, 1 / 4]
        wiesner = QuantumMoney(states, probs)
        res = wiesner.counterfeit_attack()
        self.assertEqual(np.isclose(res, 3 / 4), True)

    def test_counterfeit_attack_wiesner_money_rep_2(self):
        """Probability of counterfeit attack with 2 parallel repetitions."""
        states = [self.e_0, self.e_1, self.e_p, self.e_m]
        probs = [1 / 4, 1 / 4, 1 / 4, 1 / 4]
        reps = 2

        wiesner = QuantumMoney(states, probs, reps)
        res = wiesner.counterfeit_attack()
        self.assertEqual(np.isclose(res, (3 / 4) ** reps), True)

    def test_counterfeit_attack_wiesner_money_primal_problem(self):
        """Counterfeit attack on Wiesner's quantum money (primal problem)."""
        states = [self.e_0, self.e_1, self.e_p, self.e_m]
        probs = [1 / 4, 1 / 4, 1 / 4, 1 / 4]
        wiesner = QuantumMoney(states, probs)
        res = wiesner.primal_problem()
        self.assertEqual(np.isclose(res, 3 / 4), True)

    @pytest.mark.skip(reason="This test takes too much time.")
    def test_counterfeit_attack_wiesner_money_primal_problem_rep_2(self):
        """Counterfeit attack with 2 parallel repetitions (primal problem)."""
        states = [self.e_0, self.e_1, self.e_p, self.e_m]
        probs = [1 / 4, 1 / 4, 1 / 4, 1 / 4]
        reps = 2

        wiesner = QuantumMoney(states, probs, reps)
        res = wiesner.primal_problem()
        self.assertEqual(np.isclose(res, (3 / 4) ** reps), True)


if __name__ == "__main__":
    unittest.main()
