import numpy as np
import unittest
from toqito.nonlocal_games.extended_nonlocal_game import ExtendedNonlocalGame

class TestExtendedNonlocalGame4MUB(unittest.TestCase):
    @staticmethod
    def four_mub_game():
        """
        Define the 4-MUB extended nonlocal game (single round, qutrit bases).
        """
        d = 3
        e0, e1, e2 = np.eye(3, dtype=complex)
        zeta = np.exp(2j * np.pi / 3)
        # Four mutually unbiased bases for qutrit
        B = [
            [e0, e1, e2],
            [
                (e0 + e1 + e2) / np.sqrt(3),
                (e0 + zeta**2 * e1 + zeta * e2) / np.sqrt(3),
                (e0 + zeta * e1 + zeta**2 * e2) / np.sqrt(3),
            ],
            [
                (e0 + e1 + zeta * e2) / np.sqrt(3),
                (e0 + zeta**2 * e1 + zeta**2 * e2) / np.sqrt(3),
                (e0 + zeta * e1 + e2) / np.sqrt(3),
            ],
            [
                (e0 + e1 + zeta**2 * e2) / np.sqrt(3),
                (e0 + zeta**2 * e1 + e2) / np.sqrt(3),
                (e0 + zeta * e1 + zeta * e2) / np.sqrt(3),
            ],
        ]
        num_inputs = 4
        num_outputs = 3
        # Input distribution: uniform over x=y
        pi = np.zeros((num_inputs, num_inputs), dtype=float)
        for x in range(num_inputs):
            pi[x, x] = 1.0 / num_inputs
        # Predicate tensor: only nonzero when x=y and a=b
        pred_mat = np.zeros((d, d, num_outputs, num_outputs, num_inputs, num_inputs), dtype=complex)
        for x in range(num_inputs):
            for a in range(num_outputs):
                ket = B[x][a]
                pred_mat[:, :, a, a, x, x] = np.outer(ket, ket.conj())
        return pi, pred_mat

    def test_unentangled_value(self):
        """Classical (unentangled) value matches literature."""
        pi, pred_mat = self.four_mub_game()
        game = ExtendedNonlocalGame(pi, pred_mat)
        res = game.unentangled_value()
        expected = (3 + np.sqrt(5)) / 8
        self.assertAlmostEqual(res, expected, places=6)

    def test_quantum_lower_bound(self):
        """Quantum heuristic lower bound is close to reference value."""
        pi, pred_mat = self.four_mub_game()
        game = ExtendedNonlocalGame(pi, pred_mat)
        lb = game.quantum_value_lower_bound(initial_bob_is_random=True, seed=42, iters=50, tol=1e-6, verbose=False)
        # Reference quantum bound ≈ 0.660986 (tolerance allowed)
        self.assertAlmostEqual(lb, 0.660986, delta=5e-3)

    def test_npa_upper_bound_k1ab(self):
        """NPA upper bound at k='1+ab' approximates the quantum value."""
        pi, pred_mat = self.four_mub_game()
        game = ExtendedNonlocalGame(pi, pred_mat)
        ub = game.commuting_measurement_value_upper_bound(k="1+ab")
        # Should be close to 2/3 within tolerance
        self.assertAlmostEqual(ub, 2 / 3, delta=5e-3)

    def test_nonsignaling_value(self):
        """No-signaling value matches literature."""
        pi, pred_mat = self.four_mub_game()
        game = ExtendedNonlocalGame(pi, pred_mat)
        ns = game.nonsignaling_value()
        # Expected no-signaling value ≈ 0.788675
        self.assertAlmostEqual(ns, 0.788675, places=6)
