"""Tests for NonlocalGame class."""
import unittest
import numpy as np

from toqito.nonlocal_games.nonlocal_game import NonlocalGame


class TestNonlocalGame(unittest.TestCase):
    """Unit test for NonlocalGame."""

    def test_ffl_game_classical_value(self):
        """Classical value for the FFL game."""
        dim = 2
        num_alice_inputs, num_alice_outputs = 2, 2
        num_bob_inputs, num_bob_outputs = 2, 2
        prob_mat = np.array([[1 / 3, 1 / 3], [1 / 3, 0]])

        pred_mat = np.zeros(
            (num_alice_outputs, num_bob_outputs, num_alice_inputs, num_bob_inputs)
        )
        for a_alice in range(num_alice_outputs):
            for b_bob in range(num_bob_outputs):
                for x_alice in range(num_alice_inputs):
                    for y_bob in range(num_bob_inputs):
                        if (a_alice or x_alice) != (b_bob or y_bob):
                            pred_mat[a_alice, b_bob, x_alice, y_bob] = 1

        ffl = NonlocalGame(dim, prob_mat, pred_mat)
        res = ffl.classical_value()
        expected_res = 2 / 3
        self.assertEqual(np.isclose(res, expected_res), True)

    def test_negative_prob_mat_nonlocal_game(self):
        """Tests for invalid nonlocal game negative probability matrix."""
        with self.assertRaises(ValueError):
            dim = 2
            num_alice_inputs, num_alice_outputs = 2, 2
            num_bob_inputs, num_bob_outputs = 2, 2
            prob_mat = np.array([[1 / 4, -1 / 4], [1 / 4, 1 / 4]])

            pred_mat = np.zeros(
                (num_alice_outputs, num_bob_outputs, num_alice_inputs, num_bob_inputs)
            )
            for a_alice in range(num_alice_outputs):
                for b_bob in range(num_bob_outputs):
                    for x_alice in range(num_alice_inputs):
                        for y_bob in range(num_bob_inputs):
                            if (a_alice or x_alice) != (b_bob or y_bob):
                                pred_mat[a_alice, b_bob, x_alice, y_bob] = 1
            game = NonlocalGame(dim, prob_mat, pred_mat)
            game.classical_value()

    def test_invalid_prob_mat(self):
        """Tests for invalid probability matrix."""
        with self.assertRaises(ValueError):
            dim = 2
            num_alice_inputs, num_alice_outputs = 2, 2
            num_bob_inputs, num_bob_outputs = 2, 2
            prob_mat = np.array([[1 / 4, 1], [1 / 4, 1 / 4]])
            pred_mat = np.zeros(
                (num_alice_outputs, num_bob_outputs, num_alice_inputs, num_bob_inputs)
            )
            for a_alice in range(num_alice_outputs):
                for b_bob in range(num_bob_outputs):
                    for x_alice in range(num_alice_inputs):
                        for y_bob in range(num_bob_inputs):
                            if (a_alice or x_alice) != (b_bob or y_bob):
                                pred_mat[a_alice, b_bob, x_alice, y_bob] = 1
            game = NonlocalGame(dim, prob_mat, pred_mat)
            game.classical_value()


if __name__ == "__main__":
    unittest.main()
