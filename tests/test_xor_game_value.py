"""Tests for xor_game_value function."""
import unittest
import numpy as np

from toqito.nonlocal_games.xor_games.xor_game_value import xor_game_value


class TestXORGameValue(unittest.TestCase):
    """Unit test for xor_game_value."""

    def test_chsh_game_quantum_value(self):
        """Quantum value for the CHSH game."""
        prob_mat = np.array([[1 / 4, 1 / 4], [1 / 4, 1 / 4]])
        pred_mat = np.array([[0, 0], [0, 1]])

        res = xor_game_value(prob_mat, pred_mat, "quantum")
        expected_res = np.cos(np.pi / 8) ** 2
        self.assertEqual(np.isclose(res, expected_res), True)

    def test_chsh_game_classical_value(self):
        """Classical value for the CHSH game."""
        prob_mat = np.array([[1 / 4, 1 / 4], [1 / 4, 1 / 4]])
        pred_mat = np.array([[0, 0], [0, 1]])

        res = xor_game_value(prob_mat, pred_mat, "classical")
        expected_res = 3 / 4
        self.assertEqual(np.isclose(res, expected_res), True)

    def test_chsh_game_classical_value_tol_optimal(self):
        """Classical value for the CHSH game with optimal tolerance."""
        prob_mat = np.array([[1 / 4, 1 / 4], [1 / 4, 1 / 4]])
        pred_mat = np.array([[0, 0], [0, 1]])

        res = xor_game_value(prob_mat, pred_mat, "classical", 1)
        expected_res = 3 / 4
        self.assertEqual(np.isclose(res, expected_res), True)

    def test_odd_cycle_game_quantum_value(self):
        """Quantum value for the odd-cycle game."""
        prob_mat = np.array(
            [
                [0.1, 0.1, 0, 0, 0],
                [0, 0.1, 0.1, 0, 0],
                [0, 0, 0.1, 0.1, 0],
                [0, 0, 0, 0.1, 0.1],
                [0.1, 0, 0, 0, 0.1],
            ]
        )

        pred_mat = np.array(
            [
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0],
            ]
        )

        res = xor_game_value(prob_mat, pred_mat, "quantum")
        expected_res = 0.975528
        self.assertEqual(np.isclose(res, expected_res), True)

    def test_odd_cycle_game_classical_value(self):
        """Classical value for the odd-cycle game."""
        prob_mat = np.array(
            [
                [0.1, 0.1, 0, 0, 0],
                [0, 0.1, 0.1, 0, 0],
                [0, 0, 0.1, 0.1, 0],
                [0, 0, 0, 0.1, 0.1],
                [0.1, 0, 0, 0, 0.1],
            ]
        )

        pred_mat = np.array(
            [
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0],
            ]
        )

        res = xor_game_value(prob_mat, pred_mat, "classical")
        expected_res = 0.9
        self.assertEqual(np.isclose(res, expected_res), True)

    def test_negative_prob_mat(self):
        """Tests for invalid negative probability matrix."""
        with self.assertRaises(ValueError):
            prob_mat = np.array([[1 / 4, -1 / 4], [1 / 4, 1 / 4]])
            pred_mat = np.array([[0, 0], [0, 1]])

            xor_game_value(prob_mat, pred_mat, "quantum")

    def test_invalid_prob_mat(self):
        """Tests for invalid probability matrix."""
        with self.assertRaises(ValueError):
            prob_mat = np.array([[1 / 4, 1], [1 / 4, 1 / 4]])
            pred_mat = np.array([[0, 0], [0, 1]])

            xor_game_value(prob_mat, pred_mat, "quantum")

    def test_non_square_prob_mat(self):
        """Tests for invalid non-square probability matrix."""
        with self.assertRaises(ValueError):
            prob_mat = np.array([[1 / 4, 1 / 4, 1 / 4], [1 / 4, 1 / 4, 1 / 4]])
            pred_mat = np.array([[0, 0], [0, 1]])

            xor_game_value(prob_mat, pred_mat, "quantum")

    def test_zero_prob_mat(self):
        """Tests for zero probability matrix."""
        with self.assertRaises(ValueError):
            prob_mat = np.array([[1 / 4, 0], [1 / 4, 0]])
            pred_mat = np.array([[0, 0], [0, 1]])

            xor_game_value(prob_mat, pred_mat, "quantum")

    def test_invalid_strategy(self):
        """Tests for invalid strategy argument."""
        with self.assertRaises(ValueError):
            prob_mat = np.array([[1 / 4, 1 / 4], [1 / 4, 1 / 4]])
            pred_mat = np.array([[0, 0], [0, 1]])

            xor_game_value(prob_mat, pred_mat, "invalid")


if __name__ == "__main__":
    unittest.main()
