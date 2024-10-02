"""Tests for NonlocalGame class."""

import unittest

import numpy as np

from toqito.nonlocal_games.nonlocal_game import NonlocalGame


class TestNonlocalGame(unittest.TestCase):
    """Unit test for NonlocalGame."""

    @staticmethod
    def ffl_nonlocal_game():
        """Define the FFL nonlocal game."""
        num_alice_in, num_alice_out = 2, 2
        num_bob_in, num_bob_out = 2, 2
        prob_mat = np.array([[1 / 3, 1 / 3], [1 / 3, 0]])

        pred_mat = np.zeros((num_alice_out, num_bob_out, num_alice_in, num_bob_in))
        for a_alice in range(num_alice_out):
            for b_bob in range(num_bob_out):
                for x_alice in range(num_alice_in):
                    for y_bob in range(num_bob_in):
                        if (a_alice or x_alice) != (b_bob or y_bob):
                            pred_mat[a_alice, b_bob, x_alice, y_bob] = 1
        return prob_mat, pred_mat

    @staticmethod
    def chsh_nonlocal_game():
        """Define the CHSH nonlocal game."""
        num_alice_inputs, num_alice_outputs = 2, 2
        num_bob_inputs, num_bob_outputs = 2, 2
        prob_mat = np.array([[1 / 4, 1 / 4], [1 / 4, 1 / 4]])
        pred_mat = np.zeros((num_alice_outputs, num_bob_outputs, num_alice_inputs, num_bob_inputs))

        for a_alice in range(num_alice_outputs):
            for b_bob in range(num_bob_outputs):
                for x_alice in range(num_alice_inputs):
                    for y_bob in range(num_bob_inputs):
                        if a_alice ^ b_bob == x_alice * y_bob:
                            pred_mat[a_alice, b_bob, x_alice, y_bob] = 1
        return prob_mat, pred_mat

    @staticmethod
    def chsh_bcs_game():
        """Define the CHSH BCS game."""
        c_1 = np.zeros((2, 2))
        c_2 = np.zeros((2, 2))

        for v_1 in range(2):
            for v_2 in range(2):
                if v_1 ^ v_2 == 0:
                    c_1[v_1, v_2] = 1
                else:
                    c_2[v_1, v_2] = 1

        return [c_1, c_2]

    def test_chsh_bcs_game_to_nonlocal_game(self):
        """Conversion of BCS game to nonlocal game."""
        bcs_game = self.chsh_bcs_game()
        chsh = NonlocalGame.from_bcs_game(bcs_game)

        # Compute expected prob_mat
        prob_mat = np.array([[1 / 4, 1 / 4], [1 / 4, 1 / 4]])
        np.testing.assert_array_equal(chsh.prob_mat, prob_mat)

        # Compute expected pred_mat
        pred_mat = np.zeros((4, 2, 2, 2))
        # Compute first constraint: v1 ^ v2 = 0
        constraint1 = np.array([[1, 0], [0, 0], [0, 0], [0, 1]])
        pred_mat[:, :, 0, 0] = constraint1
        pred_mat[:, :, 0, 1] = constraint1
        # Compute second constraint: v1 ^ v2 = 1
        pred_mat[:, :, 1, 0] = np.array([[0, 0], [1, 0], [0, 1], [0, 0]])
        pred_mat[:, :, 1, 1] = np.array([[0, 0], [0, 1], [1, 0], [0, 0]])
        np.testing.assert_array_equal(chsh.pred_mat, pred_mat)

    def test_bcs_game_without_constraint(self):
        """Empty list of constraints raises exception."""
        self.assertRaises(ValueError, NonlocalGame.from_bcs_game, [])

    def test_chsh_lower_bound(self):
        """Calculate the lower bound on the quantum value for the CHSH game."""
        prob_mat, pred_mat = self.chsh_nonlocal_game()
        chsh = NonlocalGame(prob_mat, pred_mat)
        res = chsh.quantum_value_lower_bound()

        # It may be possible for the lower bound to not be equal to the correct
        # quantum value (as it may get stuck in a local minimum), but it should
        # never be possible for the lower bound to attain a value higher than
        # the true quantum value.
        self.assertLessEqual(np.isclose(res, np.cos(np.pi / 8) ** 2, rtol=1e-02), True)

        # Even with 2 qubits each, the lower bound remains the same
        res = chsh.quantum_value_lower_bound(4)
        self.assertLessEqual(np.isclose(res, np.cos(np.pi / 8) ** 2, rtol=1e-02), True)

    def test_chsh_game_classical_value(self):
        """Classical value for the CHSH game."""
        prob_mat, pred_mat = self.chsh_nonlocal_game()

        chsh = NonlocalGame(prob_mat, pred_mat)
        res = chsh.classical_value()
        expected_res = 3 / 4
        self.assertEqual(np.isclose(res, expected_res), True)

    def test_chsh_game_classical_value_rep_2(self):
        r"""Classical value for the CHSH game for 2 reps.

        Note that for classical strategies, it is known that parallel repetition
        does *not* hold for the CHSH game, that is:

        w_c(CHSH \land CHSH) = 10/16 > 9/16 = w_c(CHSH) w_c(CHSH).
        """
        prob_mat, pred_mat = self.chsh_nonlocal_game()

        chsh = NonlocalGame(prob_mat, pred_mat, 2)
        res = chsh.classical_value()
        expected_res = 10 / 16
        self.assertEqual(np.isclose(res, expected_res), True)

    def test_bcs_chsh_lower_bound(self):
        """Calculate the lower bound on the quantum value for the converted BCS CHSH game."""
        bcs_game = self.chsh_bcs_game()
        chsh = NonlocalGame.from_bcs_game(bcs_game)
        res = chsh.quantum_value_lower_bound()
        self.assertLessEqual(np.isclose(res, np.cos(np.pi / 8) ** 2, rtol=1e-02), True)

    def test_bcs_chsh_game_classical_value(self):
        """Classical value for the converted BCS CHSH game."""
        bcs_game = self.chsh_bcs_game()
        chsh = NonlocalGame.from_bcs_game(bcs_game)
        res = chsh.classical_value()
        expected_res = 3 / 4
        self.assertEqual(np.isclose(res, expected_res), True)

    def test_bcs_chsh_game_classical_value_rep_2(self):
        r"""Classical value for the CHSH game for 2 reps.

        Note that for classical strategies, it is known that parallel repetition
        does *not* hold for the CHSH game, that is:

        w_c(CHSH \land CHSH) = 10/16 > 9/16 = w_c(CHSH) w_c(CHSH).
        """
        bcs_game = self.chsh_bcs_game()
        chsh = NonlocalGame.from_bcs_game(bcs_game, 2)
        res = chsh.classical_value()
        expected_res = 10 / 16
        self.assertEqual(np.isclose(res, expected_res), True)

    def test_ffl_game_classical_value(self):
        """Classical value for the FFL game."""
        prob_mat, pred_mat = self.ffl_nonlocal_game()

        ffl = NonlocalGame(prob_mat, pred_mat)
        res = ffl.classical_value()
        expected_res = 2 / 3
        self.assertEqual(np.isclose(res, expected_res), True)

    def test_ffl_game_classical_value_rep_2(self):
        """Classical value for the FFL game for 2 reps."""
        prob_mat, pred_mat = self.ffl_nonlocal_game()

        ffl = NonlocalGame(prob_mat, pred_mat, 2)
        res = ffl.classical_value()
        expected_res = 2 / 3
        self.assertEqual(np.isclose(res, expected_res), True)

    def test_ffl_game_quantum_value_lower_bound_value_rep_2(self):
        """Lower bound on quantum value for the FFL game for 2 reps."""
        prob_mat, pred_mat = self.ffl_nonlocal_game()

        ffl = NonlocalGame(prob_mat, pred_mat, 2)
        res = ffl.quantum_value_lower_bound()
        expected_res = 2 / 3
        self.assertLessEqual(np.isclose(res, expected_res), True)

    def test_ffl_game_nonsignaling_value(self):
        """Non-signaling value for the FFL game."""
        prob_mat, pred_mat = self.ffl_nonlocal_game()

        ffl = NonlocalGame(prob_mat, pred_mat)
        res = ffl.nonsignaling_value()
        expected_res = 2 / 3
        self.assertEqual(np.isclose(res, expected_res, atol=0.5), True)

    def test_ffl_game_nonsignaling_value_rep_2(self):
        """Non-signaling value for the FFL game for 2 reps."""
        prob_mat, pred_mat = self.ffl_nonlocal_game()

        ffl = NonlocalGame(prob_mat, pred_mat, 2)
        res = ffl.nonsignaling_value()
        expected_res = 2 / 3
        self.assertEqual(np.isclose(res, expected_res, atol=0.5), True)

    def test_chsh_game_nonsignaling_value(self):
        """Non-signaling value for the CHSH game."""
        prob_mat, pred_mat = self.chsh_nonlocal_game()

        chsh = NonlocalGame(prob_mat, pred_mat)
        res = chsh.nonsignaling_value()
        expected_res = 1
        self.assertEqual(np.isclose(res, expected_res, atol=0.5), True)

    def test_chsh_game_nonsignaling_value_rep_2(self):
        """Non-signaling value for the CHSH game for 2 reps."""
        prob_mat, pred_mat = self.chsh_nonlocal_game()

        chsh = NonlocalGame(prob_mat, pred_mat, 2)
        res = chsh.nonsignaling_value()
        expected_res = 1
        self.assertEqual(np.isclose(res, expected_res, atol=0.5), True)

    def test_chsh_game_commuting_measurement_value(self):
        """Commuting measurement value for the CHSH game."""
        prob_mat, pred_mat = self.chsh_nonlocal_game()

        chsh = NonlocalGame(prob_mat, pred_mat)
        res = chsh.commuting_measurement_value_upper_bound(k=1)
        expected_res = 0.8535
        self.assertEqual(np.isclose(res, expected_res, atol=0.5), True)

    def test_ffl_game_commuting_measurement_value(self):
        """Commuting measurement value for the FFL game."""
        prob_mat, pred_mat = self.ffl_nonlocal_game()

        ffl = NonlocalGame(prob_mat, pred_mat)
        res = ffl.commuting_measurement_value_upper_bound(k=1)
        expected_res = 0.666
        self.assertEqual(np.isclose(res, expected_res, atol=0.5), True)

    def test_unbalanced_nonlocal_game(self):
        """Test nonlocal game where Bob has more outputs than Alice."""
        num_alice_inputs, num_alice_outputs = 2, 1
        num_bob_inputs, num_bob_outputs = 2, 3

        prob_mat = np.array([[1 / 6, 1 / 6, 1 / 6], [1 / 6, 1 / 6, 1 / 6]])
        pred_mat = np.zeros((num_alice_outputs, num_bob_outputs, num_alice_inputs, num_bob_inputs))

        for a_alice in range(num_alice_outputs):
            for b_bob in range(num_bob_outputs):
                for x_alice in range(num_alice_inputs):
                    for y_bob in range(num_bob_inputs):
                        if a_alice ^ b_bob == x_alice * y_bob:
                            pred_mat[a_alice, b_bob, x_alice, y_bob] = 1

        game = NonlocalGame(prob_mat, pred_mat)
        res = game.classical_value()

        # Expected result: We do not care for the value, just triggering the block
        self.assertIsNotNone(res)
