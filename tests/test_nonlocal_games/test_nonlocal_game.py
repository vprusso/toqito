"""Tests for NonlocalGame class."""
import unittest
import numpy as np

from toqito.nonlocal_games.nonlocal_game import NonlocalGame


class TestNonlocalGame(unittest.TestCase):
    """Unit test for NonlocalGame."""

    def test_ffl_game_classical_value(self):
        """Classical value for the FFL game."""
        dim = 2
        num_alice_in, num_alice_out = 2, 2
        num_bob_in, num_bob_out = 2, 2
        prob_mat = np.array([[1 / 3, 1 / 3], [1 / 3, 0]])

        pred_mat = np.zeros(
            (num_alice_out, num_bob_out, num_alice_in, num_bob_in)
        )
        for a_alice in range(num_alice_out):
            for b_bob in range(num_bob_out):
                for x_alice in range(num_alice_in):
                    for y_bob in range(num_bob_in):
                        if (a_alice or x_alice) != (b_bob or y_bob):
                            pred_mat[a_alice, b_bob, x_alice, y_bob] = 1

        ffl = NonlocalGame(dim, prob_mat, pred_mat)
        res = ffl.classical_value()
        expected_res = 2 / 3
        self.assertEqual(np.isclose(res, expected_res), True)


if __name__ == "__main__":
    unittest.main()
