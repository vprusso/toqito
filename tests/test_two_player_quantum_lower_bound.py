"""Tests for nonlocal_game_lower_bound function."""
import unittest
import numpy as np

from toqito.nonlocal_games.two_player_quantum_lower_bound import (
    two_player_quantum_lower_bound,
)


class TestNonlocalGameLowerBound(unittest.TestCase):
    """Unit test for nonlocal_game_lower_bound."""

    def test_chsh_lower_bound(self):
        """Calculate the lower bound on the quantum value for the CHSH game."""
        dim = 2
        num_alice_inputs, num_alice_outputs = 2, 2
        num_bob_inputs, num_bob_outputs = 2, 2
        prob_mat = np.array([[1 / 4, 1 / 4], [1 / 4, 1 / 4]])
        pred_mat = np.zeros(
            (num_alice_outputs, num_bob_outputs, num_alice_inputs, num_bob_inputs)
        )

        for a_alice in range(num_alice_outputs):
            for b_bob in range(num_bob_outputs):
                for x_alice in range(num_alice_inputs):
                    for y_bob in range(num_bob_inputs):
                        if np.mod(a_alice + b_bob + x_alice * y_bob, dim) == 0:
                            pred_mat[a_alice, b_bob, x_alice, y_bob] = 1

        lower_bound = two_player_quantum_lower_bound(dim, prob_mat, pred_mat)
        self.assertEqual(
            np.isclose(lower_bound, np.cos(np.pi / 8) ** 2, rtol=1e-02), True
        )


if __name__ == "__main__":
    unittest.main()
