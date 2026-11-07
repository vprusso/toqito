"""Tests for NonlocalGame class."""

import itertools
import re
import unittest
from unittest import mock

import cvxpy
import numpy as np
import pytest

from toqito.nonlocal_games.nonlocal_game import NonlocalGame, _deterministic_strategies


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
        with pytest.raises(ValueError, match="At least 1 constraint is required"):
            NonlocalGame.from_bcs_game([])

    def test_bcs_game_degenerate_constraint(self):
        """A constraint with no dependent variables raises exception."""
        # A constant constraint tensor does not depend on any variable.
        constraint = np.ones((2, 2))
        with pytest.raises(ValueError, match=re.escape("Constraint 0 is degenerate (has no dependent variables).")):
            NonlocalGame.from_bcs_game([constraint])

    def test_bcs_game_malformed_constraint_shape(self):
        """Constraints must all be truth tables of shape (2,) * n."""
        with pytest.raises(ValueError, match="truth table of shape"):
            NonlocalGame.from_bcs_game([np.array([1, 1, 0])])
        with pytest.raises(ValueError, match="truth table of shape"):
            NonlocalGame.from_bcs_game([np.eye(2), np.ones((2, 2, 2))])

    def test_invalid_game_inputs(self):
        """The constructor rejects mismatched shapes and improper question distributions."""
        prob_mat, pred_mat = self.chsh_nonlocal_game()
        with pytest.raises(ValueError, match="must have 4 dimensions"):
            NonlocalGame(prob_mat, pred_mat[0])
        with pytest.raises(ValueError, match="question pairs"):
            NonlocalGame(np.full((2, 3), 1 / 6), pred_mat)
        with pytest.raises(ValueError, match="non-negative"):
            NonlocalGame(np.array([[1 / 2, 1 / 2], [1 / 2, -1 / 2]]), pred_mat)
        with pytest.raises(ValueError, match="sum to 1"):
            NonlocalGame(np.full((2, 2), 1 / 2), pred_mat)

    def test_chsh_lower_bound(self):
        """The see-saw reaches the quantum value of the CHSH game from random starts."""
        prob_mat, pred_mat = self.chsh_nonlocal_game()
        chsh = NonlocalGame(prob_mat, pred_mat)
        res = chsh.quantum_value_lower_bound(seed=0)
        self.assertAlmostEqual(res, np.cos(np.pi / 8) ** 2, delta=1e-4)

        # Larger local dimension cannot help for CHSH, and the bound stays achievable.
        res = chsh.quantum_value_lower_bound(dim=4, seed=0)
        self.assertLessEqual(res, np.cos(np.pi / 8) ** 2 + 1e-4)
        self.assertGreater(res, 3 / 4 + 1e-3)

    def test_chsh_lower_bound_converges_early(self):
        """With a loose tolerance the see-saw stops after one step and still returns an achievable value."""
        prob_mat, pred_mat = self.chsh_nonlocal_game()
        chsh = NonlocalGame(prob_mat, pred_mat)
        res = chsh.quantum_value_lower_bound(tol=1e-1, seed=42)
        self.assertGreater(res, 3 / 4 - 1e-6)
        self.assertLessEqual(res, np.cos(np.pi / 8) ** 2 + 1e-4)

    def test_chsh_lower_bound_runs_to_step_cap(self):
        """With a tolerance the convergence check can never meet, the see-saw runs `max_steps` steps and returns."""
        prob_mat, pred_mat = self.chsh_nonlocal_game()
        chsh = NonlocalGame(prob_mat, pred_mat)
        res = chsh.quantum_value_lower_bound(iters=1, tol=0.0, seed=1, max_steps=3)
        self.assertGreater(res, 3 / 4 - 1e-6)
        self.assertLessEqual(res, np.cos(np.pi / 8) ** 2 + 1e-4)

    def test_chsh_lower_bound_seed_reproducible(self):
        """A fixed seed makes the see-saw lower bound reproducible and a valid bound."""
        prob_mat, pred_mat = self.chsh_nonlocal_game()
        chsh = NonlocalGame(prob_mat, pred_mat)

        res1 = chsh.quantum_value_lower_bound(seed=42)
        res2 = chsh.quantum_value_lower_bound(seed=42)

        self.assertTrue(np.isclose(res1, res2))
        self.assertAlmostEqual(res1, np.cos(np.pi / 8) ** 2, delta=1e-4)

    def test_lower_bound_raises_when_solver_fails(self):
        """A see-saw step whose SDP does not solve is reported rather than silently skipped."""
        prob_mat, pred_mat = self.chsh_nonlocal_game()
        chsh = NonlocalGame(prob_mat, pred_mat)
        with mock.patch.object(cvxpy.Problem, "solve", lambda self, *args, **kwargs: None):
            with pytest.raises(ValueError, match="did not solve successfully"):
                chsh.quantum_value_lower_bound(iters=1)

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
        """The see-saw reaches the quantum value of the converted BCS CHSH game."""
        bcs_game = self.chsh_bcs_game()
        chsh = NonlocalGame.from_bcs_game(bcs_game)
        res = chsh.quantum_value_lower_bound(seed=0)
        self.assertAlmostEqual(res, np.cos(np.pi / 8) ** 2, delta=1e-4)

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

    @pytest.mark.slow
    def test_ffl_game_quantum_value_lower_bound_value_rep_2(self):
        """Lower bound on quantum value for the FFL game for 2 reps."""
        prob_mat, pred_mat = self.ffl_nonlocal_game()

        ffl = NonlocalGame(prob_mat, pred_mat, 2)
        res = ffl.quantum_value_lower_bound(seed=0)
        # Two parallel repetitions of FFL still have quantum value 2/3, and the see-saw attains it.
        self.assertAlmostEqual(res, 2 / 3, delta=1e-4)

    def test_ffl_game_nonsignaling_value(self):
        """Non-signaling value for the FFL game."""
        prob_mat, pred_mat = self.ffl_nonlocal_game()

        ffl = NonlocalGame(prob_mat, pred_mat)
        res = ffl.nonsignaling_value()
        expected_res = 2 / 3
        self.assertEqual(np.isclose(res, expected_res, atol=1e-3), True)

    def test_ffl_game_nonsignaling_value_rep_2(self):
        """Non-signaling value for the FFL game for 2 reps."""
        prob_mat, pred_mat = self.ffl_nonlocal_game()

        ffl = NonlocalGame(prob_mat, pred_mat, 2)
        res = ffl.nonsignaling_value()
        expected_res = 2 / 3
        self.assertEqual(np.isclose(res, expected_res, atol=1e-3), True)

    def test_chsh_game_nonsignaling_value(self):
        """Non-signaling value for the CHSH game."""
        prob_mat, pred_mat = self.chsh_nonlocal_game()

        chsh = NonlocalGame(prob_mat, pred_mat)
        res = chsh.nonsignaling_value()
        expected_res = 1
        self.assertEqual(np.isclose(res, expected_res, atol=1e-3), True)

    def test_chsh_game_nonsignaling_value_rep_2(self):
        """Non-signaling value for the CHSH game for 2 reps."""
        prob_mat, pred_mat = self.chsh_nonlocal_game()

        chsh = NonlocalGame(prob_mat, pred_mat, 2)
        res = chsh.nonsignaling_value()
        expected_res = 1
        self.assertEqual(np.isclose(res, expected_res, atol=1e-3), True)

    def test_chsh_game_commuting_measurement_value(self):
        """Commuting measurement value for the CHSH game."""
        prob_mat, pred_mat = self.chsh_nonlocal_game()

        chsh = NonlocalGame(prob_mat, pred_mat)
        res = chsh.commuting_measurement_value_upper_bound(k=1)
        expected_res = 0.8535
        self.assertEqual(np.isclose(res, expected_res, atol=1e-3), True)

    def test_ffl_game_commuting_measurement_value(self):
        """Commuting measurement value for the FFL game."""
        prob_mat, pred_mat = self.ffl_nonlocal_game()

        ffl = NonlocalGame(prob_mat, pred_mat)
        res = ffl.commuting_measurement_value_upper_bound(k=1)
        expected_res = 0.666
        self.assertEqual(np.isclose(res, expected_res, atol=1e-3), True)

    def test_unbalanced_nonlocal_game(self):
        """Test nonlocal game where Bob has more outputs than Alice."""
        num_alice_inputs, num_alice_outputs = 2, 1
        num_bob_inputs, num_bob_outputs = 2, 3

        prob_mat = np.full((num_alice_inputs, num_bob_inputs), 1 / 4)
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

    def test_chsh_bcs_game_classical_value(self):
        """Classical value for the CHSH game (fast method)."""
        c1 = np.zeros((2, 2))
        c2 = np.zeros((2, 2))
        for v1 in range(2):
            for v2 in range(2):
                (c1 if (v1 ^ v2) == 0 else c2)[v1, v2] = 1
        game = NonlocalGame.from_bcs_game([c1, c2])
        val = game.classical_value()
        self.assertAlmostEqual(val, 0.75, places=6)

    def test_four_line_parity_bcs_classical_value(self):
        """Classical value for the 4-Line Parity BCS game."""
        shape = (2, 2, 2, 2, 2, 2)
        c_1 = np.zeros(shape)
        c_2 = np.zeros(shape)
        c_3 = np.zeros(shape)
        c_4 = np.zeros(shape)

        for v1 in range(2):
            for v2 in range(2):
                for v3 in range(2):
                    for v4 in range(2):
                        for v5 in range(2):
                            for v6 in range(2):
                                if (v1 + v2 + v3) % 2 == 0:
                                    c_1[v1, v2, v3, v4, v5, v6] = 1
                                if (v3 + v4 + v5) % 2 == 0:
                                    c_2[v1, v2, v3, v4, v5, v6] = 1
                                if (v5 + v6 + v1) % 2 == 0:
                                    c_3[v1, v2, v3, v4, v5, v6] = 1
                                if (v2 + v4 + v6) % 2 == 1:
                                    c_4[v1, v2, v3, v4, v5, v6] = 1

        game = NonlocalGame.from_bcs_game([c_1, c_2, c_3, c_4])
        val = game.classical_value()
        self.assertAlmostEqual(val, 11 / 12, places=6)

    def test_mermin_peres_bcs_classical_value(self):
        """Classical value for the Mermin–Peres BCS game."""
        shape = (2,) * 9
        c_1 = np.zeros(shape)
        c_2 = np.zeros(shape)
        c_3 = np.zeros(shape)
        c_4 = np.zeros(shape)
        c_5 = np.zeros(shape)
        c_6 = np.zeros(shape)

        for v1 in range(2):
            for v2 in range(2):
                for v3 in range(2):
                    for v4 in range(2):
                        for v5 in range(2):
                            for v6 in range(2):
                                for v7 in range(2):
                                    for v8 in range(2):
                                        for v9 in range(2):
                                            if v1 ^ v2 ^ v3 == 0:
                                                c_1[v1, v2, v3] = 1
                                            if v4 ^ v5 ^ v6 == 0:
                                                c_2[v4, v5, v6] = 1
                                            if v7 ^ v8 ^ v9 == 0:
                                                c_3[v7, v8, v9] = 1
                                            if v1 ^ v4 ^ v7 == 0:
                                                c_4[v1, v4, v7] = 1
                                            if v2 ^ v5 ^ v8 == 0:
                                                c_5[v2, v5, v8] = 1
                                            if v3 ^ v6 ^ v9 == 1:
                                                c_6[v3, v6, v9] = 1

        game = NonlocalGame.from_bcs_game([c_1, c_2, c_3, c_4, c_5, c_6])
        val = game.classical_value()
        self.assertAlmostEqual(val, 17 / 18, places=6)

    def test_three_bit_parity_bcs_classical_value(self):
        """Classical value for the 3-bit parity BCS game."""
        c_1 = np.zeros((2, 2, 2))
        c_2 = np.zeros((2, 2, 2))

        for v1 in range(2):
            for v2 in range(2):
                for v3 in range(2):
                    if (v1 + v2 + v3) % 2 == 0:
                        c_1[v1, v2, v3] = 1
                    else:
                        c_2[v1, v2, v3] = 1

        game = NonlocalGame.from_bcs_game([c_1, c_2])
        val = game.classical_value()
        self.assertAlmostEqual(val, 5 / 6, places=6)


class TestNonlocalGameStrategies(unittest.TestCase):
    """The strategies returned alongside each value reproduce that value."""

    @staticmethod
    def chsh():
        """Build the CHSH game."""
        return NonlocalGame(*TestNonlocalGame.chsh_nonlocal_game())

    def test_classical_strategy_attains_classical_value(self):
        """The optimal deterministic strategy wins with the classical value."""
        chsh = self.chsh()
        value, (alice_outputs, bob_outputs) = chsh.classical_value(return_strategy=True)
        self.assertAlmostEqual(value, 3 / 4)
        self.assertEqual(alice_outputs.shape, (2,))
        self.assertEqual(bob_outputs.shape, (2,))
        self.assertAlmostEqual(chsh.classical_strategy_value(alice_outputs, bob_outputs), value)

    def test_classical_value_chunked_enumeration_matches(self):
        """Enumerating Bob's strategies in several chunks gives the same value and strategy as one chunk."""
        rng = np.random.default_rng(7)
        prob_mat = np.full((3, 4), 1 / 12)
        pred_mat = (rng.random((3, 3, 3, 4)) < 0.5).astype(float)
        game = NonlocalGame(prob_mat, pred_mat)
        value, (alice_outputs, bob_outputs) = game.classical_value(return_strategy=True)
        # Bob has 3**4 = 81 strategies; a budget of one strategy's worth of bytes forces 81 chunks.
        with mock.patch("toqito.nonlocal_games.nonlocal_game._STRATEGY_CHUNK_BYTES", 1):
            chunked_value, (chunked_alice, chunked_bob) = game.classical_value(return_strategy=True)
        self.assertAlmostEqual(chunked_value, value)
        np.testing.assert_array_equal(chunked_alice, alice_outputs)
        np.testing.assert_array_equal(chunked_bob, bob_outputs)
        self.assertAlmostEqual(game.classical_strategy_value(alice_outputs, bob_outputs), value)

    def test_deterministic_strategies_are_numbered_lexicographically(self):
        """Strategy ``s`` is the ``s``-th answer tuple in lexicographic order, as `itertools.product` lists them."""
        expected = np.array(list(itertools.product(range(3), repeat=4)))
        np.testing.assert_array_equal(_deterministic_strategies(3, 4, 0, 81), expected)
        np.testing.assert_array_equal(_deterministic_strategies(3, 4, 10, 25), expected[10:25])

    def test_classical_strategy_when_roles_are_swapped(self):
        """The strategy is reported in the right order when Alice's strategy space is the smaller one."""
        prob_mat = np.full((2, 2), 1 / 4)
        pred_mat = np.zeros((1, 3, 2, 2))
        pred_mat[0, 1, 0, :] = 1
        pred_mat[0, 2, 1, :] = 1
        game = NonlocalGame(prob_mat, pred_mat)
        value, (alice_outputs, bob_outputs) = game.classical_value(return_strategy=True)
        self.assertEqual(alice_outputs.shape, (2,))
        self.assertEqual(bob_outputs.shape, (2,))
        self.assertAlmostEqual(game.classical_strategy_value(alice_outputs, bob_outputs), value)
        self.assertAlmostEqual(value, 1 / 2)

    def test_quantum_strategy_attains_lower_bound(self):
        """The returned state and measurements form a valid strategy whose winning probability is the bound."""
        chsh = self.chsh()
        value, strategy = chsh.quantum_value_lower_bound(seed=0, return_strategy=True)
        self.assertAlmostEqual(value, np.cos(np.pi / 8) ** 2, delta=1e-4)
        self.assertAlmostEqual(chsh.quantum_strategy_value(strategy), value, delta=1e-6)

        self.assertEqual(strategy.state.shape, (4, 4))
        self.assertAlmostEqual(np.trace(strategy.state).real, 1.0, delta=1e-8)
        self.assertGreaterEqual(np.linalg.eigvalsh(strategy.state).min(), -1e-8)
        for povms in (strategy.alice_povms, strategy.bob_povms):
            self.assertEqual(povms.shape, (2, 2, 2, 2))
            np.testing.assert_allclose(
                povms.sum(axis=3), np.broadcast_to(np.eye(2)[:, :, np.newaxis], (2, 2, 2)), atol=1e-6
            )
            self.assertGreaterEqual(np.linalg.eigvalsh(np.moveaxis(povms, (2, 3), (0, 1))).min(), -1e-6)

    def test_quantum_strategy_with_larger_dimension(self):
        """A strategy on larger local systems is reconstructed on the right dimensions."""
        chsh = self.chsh()
        value, strategy = chsh.quantum_value_lower_bound(dim=3, iters=1, seed=0, return_strategy=True)
        self.assertEqual(strategy.state.shape, (9, 9))
        self.assertEqual(strategy.alice_povms.shape, (3, 3, 2, 2))
        self.assertAlmostEqual(chsh.quantum_strategy_value(strategy), value, delta=1e-6)

    def test_nonsignaling_distribution_attains_value(self):
        """The optimal non-signaling distribution is normalised, non-signaling and wins with the value."""
        chsh = self.chsh()
        value, distribution = chsh.nonsignaling_value(return_strategy=True)
        self.assertAlmostEqual(value, 1.0, delta=1e-6)
        self.assertAlmostEqual(chsh.distribution_value(distribution), value, delta=1e-6)
        np.testing.assert_allclose(distribution.sum(axis=(0, 1)), 1.0, atol=1e-6)
        alice_marginal = distribution.sum(axis=1)
        np.testing.assert_allclose(alice_marginal[:, :, 0], alice_marginal[:, :, 1], atol=1e-6)

    def test_commuting_distribution_attains_upper_bound(self):
        """The distribution of the level-1 relaxation wins with the upper bound."""
        chsh = self.chsh()
        value, distribution = chsh.commuting_measurement_value_upper_bound(k=1, return_strategy=True)
        self.assertAlmostEqual(value, np.cos(np.pi / 8) ** 2, delta=1e-4)
        self.assertAlmostEqual(chsh.distribution_value(distribution), value, delta=1e-6)
        np.testing.assert_allclose(distribution.sum(axis=(0, 1)), 1.0, atol=1e-5)
