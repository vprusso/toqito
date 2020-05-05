"""Tests for counterfeit_attack function."""
import unittest
import numpy as np

from toqito.core.ket import ket
from toqito.states.operations.tensor import tensor
from toqito.nonlocal_games.quantum_money.counterfeit_attack import counterfeit_attack


class TestCounterfeitAttack(unittest.TestCase):
    """Unit test for counterfeit_attack."""

    def test_counterfeit_attack_wiesner_money(self):
        """Probability of counterfeit attack on Wiesner's quantum money."""
        e_0, e_1 = ket(2, 0), ket(2, 1)
        e_p = (e_0 + e_1) / np.sqrt(2)
        e_m = (e_0 - e_1) / np.sqrt(2)

        e_000 = tensor(e_0, e_0, e_0)
        e_111 = tensor(e_1, e_1, e_1)
        e_ppp = tensor(e_p, e_p, e_p)
        e_mmm = tensor(e_m, e_m, e_m)

        q_a = (
            1
            / 4
            * (
                e_000 * e_000.conj().T
                + e_111 * e_111.conj().T
                + e_ppp * e_ppp.conj().T
                + e_mmm * e_mmm.conj().T
            )
        )
        res = counterfeit_attack(q_a)
        self.assertEqual(np.isclose(res, 3 / 4), True)


if __name__ == "__main__":
    unittest.main()
