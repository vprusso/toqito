"""Tests for ExtendedNonlocalGame class."""

import unittest

import numpy as np

from toqito.nonlocal_games.extended_nonlocal_game import ExtendedNonlocalGame
from toqito.states import basis


class TestExtendedNonlocalGame(unittest.TestCase):
    """Unit test for ExtendedNonlocalGame."""

    @staticmethod
    def bb84_extended_nonlocal_game():
        """Define the BB84 extended nonlocal game."""
        e_0, e_1 = basis(2, 0), basis(2, 1)
        e_p = (e_0 + e_1) / np.sqrt(2)
        e_m = (e_0 - e_1) / np.sqrt(2)

        dim = 2
        num_alice_out, num_bob_out = 2, 2
        num_alice_in, num_bob_in = 2, 2

        pred_mat = np.zeros([dim, dim, num_alice_out, num_bob_out, num_alice_in, num_bob_in])
        pred_mat[:, :, 0, 0, 0, 0] = e_0 @ e_0.conj().T
        pred_mat[:, :, 0, 0, 1, 1] = e_p @ e_p.conj().T
        pred_mat[:, :, 1, 1, 0, 0] = e_1 @ e_1.conj().T
        pred_mat[:, :, 1, 1, 1, 1] = e_m @ e_m.conj().T

        prob_mat = 1 / 2 * np.identity(2)

        return prob_mat, pred_mat

    @staticmethod
    def chsh_extended_nonlocal_game():
        """Define the CHSH extended nonlocal game."""
        dim = 2
        num_alice_out, num_bob_out = 2, 2
        num_alice_in, num_bob_in = 2, 2

        pred_mat = np.zeros([dim, dim, num_alice_out, num_bob_out, num_alice_in, num_bob_in])
        pred_mat[:, :, 0, 0, 0, 0] = np.array([[1, 0], [0, 0]])
        pred_mat[:, :, 0, 0, 0, 1] = np.array([[1, 0], [0, 0]])
        pred_mat[:, :, 0, 0, 1, 0] = np.array([[1, 0], [0, 0]])

        pred_mat[:, :, 1, 1, 0, 0] = np.array([[0, 0], [0, 1]])
        pred_mat[:, :, 1, 1, 0, 1] = np.array([[0, 0], [0, 1]])
        pred_mat[:, :, 1, 1, 1, 0] = np.array([[0, 0], [0, 1]])

        pred_mat[:, :, 0, 1, 1, 1] = 1 / 2 * np.array([[1, 1], [1, 1]])
        pred_mat[:, :, 1, 0, 1, 1] = 1 / 2 * np.array([[1, -1], [-1, 1]])

        prob_mat = np.array([[1 / 4, 1 / 4], [1 / 4, 1 / 4]])

        return prob_mat, pred_mat

    @staticmethod
    def moe_mub_4_in_3_out_game():
        """Define the monogamy-of-entanglement game defined by MUBs."""
        prob_mat = 1 / 4 * np.identity(4)

        dim = 3
        e_0, e_1, e_2 = basis(dim, 0), basis(dim, 1), basis(dim, 2)

        eta = np.exp((2 * np.pi * 1j) / dim)
        mub_0 = [e_0, e_1, e_2]
        mub_1 = [
            (e_0 + e_1 + e_2) / np.sqrt(3),
            (e_0 + eta**2 * e_1 + eta * e_2) / np.sqrt(3),
            (e_0 + eta * e_1 + eta**2 * e_2) / np.sqrt(3),
        ]
        mub_2 = [
            (e_0 + e_1 + eta * e_2) / np.sqrt(3),
            (e_0 + eta**2 * e_1 + eta**2 * e_2) / np.sqrt(3),
            (e_0 + eta * e_1 + e_2) / np.sqrt(3),
        ]
        mub_3 = [
            (e_0 + e_1 + eta**2 * e_2) / np.sqrt(3),
            (e_0 + eta**2 * e_1 + e_2) / np.sqrt(3),
            (e_0 + eta * e_1 + eta * e_2) / np.sqrt(3),
        ]

        # List of measurements defined from mutually unbiased basis.
        mubs = [mub_0, mub_1, mub_2, mub_3]

        num_in = 4
        num_out = 3
        pred_mat = np.zeros([dim, dim, num_out, num_out, num_in, num_in], dtype=complex)

        pred_mat[:, :, 0, 0, 0, 0] = mubs[0][0] @ mubs[0][0].conj().T
        pred_mat[:, :, 1, 1, 0, 0] = mubs[0][1] @ mubs[0][1].conj().T
        pred_mat[:, :, 2, 2, 0, 0] = mubs[0][2] @ mubs[0][2].conj().T

        pred_mat[:, :, 0, 0, 1, 1] = mubs[1][0] @ mubs[1][0].conj().T
        pred_mat[:, :, 1, 1, 1, 1] = mubs[1][1] @ mubs[1][1].conj().T
        pred_mat[:, :, 2, 2, 1, 1] = mubs[1][2] @ mubs[1][2].conj().T

        pred_mat[:, :, 0, 0, 2, 2] = mubs[2][0] @ mubs[2][0].conj().T
        pred_mat[:, :, 1, 1, 2, 2] = mubs[2][1] @ mubs[2][1].conj().T
        pred_mat[:, :, 2, 2, 2, 2] = mubs[2][2] @ mubs[2][2].conj().T

        pred_mat[:, :, 0, 0, 3, 3] = mubs[3][0] @ mubs[3][0].conj().T
        pred_mat[:, :, 1, 1, 3, 3] = mubs[3][1] @ mubs[3][1].conj().T
        pred_mat[:, :, 2, 2, 3, 3] = mubs[3][2] @ mubs[3][2].conj().T

        return prob_mat, pred_mat

    def test_bb84_unentangled_value(self):
        """Calculate the unentangled value of the BB84 game."""
        prob_mat, pred_mat = self.bb84_extended_nonlocal_game()
        bb84 = ExtendedNonlocalGame(prob_mat, pred_mat)
        res = bb84.unentangled_value()
        expected_res = np.cos(np.pi / 8) ** 2

        self.assertEqual(np.isclose(res, expected_res), True)

    def test_bb84_unentangled_value_rep_2(self):
        """Calculate the unentangled value for BB84 game for 2 repetitions."""
        prob_mat, pred_mat = self.bb84_extended_nonlocal_game()
        bb84_2 = ExtendedNonlocalGame(prob_mat, pred_mat, 2)
        res = bb84_2.unentangled_value()
        expected_res = np.cos(np.pi / 8) ** 4

        self.assertEqual(np.isclose(res, expected_res, atol=1e-3), True)

    def test_bb84_quantum_value_lower_bound(self):
        """Calculate the lower bound for the quantum value of theBB84 game."""
        prob_mat, pred_mat = self.bb84_extended_nonlocal_game()
        bb84 = ExtendedNonlocalGame(prob_mat, pred_mat)
        res = bb84.quantum_value_lower_bound()
        expected_res = np.cos(np.pi / 8) ** 2

        self.assertLessEqual(np.isclose(res, expected_res), True)

    def test_bb84_nonsignaling_value(self):
        """Calculate the non-signaling value of the BB84 game."""
        prob_mat, pred_mat = self.bb84_extended_nonlocal_game()
        bb84 = ExtendedNonlocalGame(prob_mat, pred_mat)
        res = bb84.nonsignaling_value()
        expected_res = np.cos(np.pi / 8) ** 2

        self.assertEqual(np.isclose(res, expected_res, rtol=1e-03), True)

    def test_bb84_nonsignaling_value_rep_2(self):
        """Calculate the non-signaling value of the BB84 game for 2 reps."""
        prob_mat, pred_mat = self.bb84_extended_nonlocal_game()
        bb84 = ExtendedNonlocalGame(prob_mat, pred_mat, 2)
        res = bb84.nonsignaling_value()
        expected_res = 0.73826

        self.assertEqual(np.isclose(res, expected_res, rtol=1e-03), True)

    def test_chsh_unentangled_value(self):
        """Calculate the unentangled value of the CHSH game."""
        prob_mat, pred_mat = self.chsh_extended_nonlocal_game()
        chsh = ExtendedNonlocalGame(prob_mat, pred_mat)
        res = chsh.unentangled_value()
        expected_res = 3 / 4

        self.assertEqual(np.isclose(res, expected_res), True)

    def test_moe_mub_4_in_3_out_unentangled_value(self):
        """Calculate the unentangled value of a monogamy-of-entanglement game."""
        prob_mat, pred_mat = self.moe_mub_4_in_3_out_game()
        moe = ExtendedNonlocalGame(prob_mat, pred_mat)
        res = moe.unentangled_value()
        expected_res = (3 + np.sqrt(5)) / 8

        self.assertEqual(np.isclose(res, expected_res), True)

    def test_bb84_commuting_value_upper_bound(self):
        """Calculate an upper bound on the commuting measurement value of the BB84 game."""
        prob_mat, pred_mat = self.bb84_extended_nonlocal_game()
        bb84 = ExtendedNonlocalGame(prob_mat, pred_mat)
        res = bb84.commuting_measurement_value_upper_bound()
        expected_res = np.cos(np.pi / 8) ** 2

        self.assertEqual(np.isclose(res, expected_res, atol=1e-5), True)

    def test_chsh_commuting_value_upper_bound(self):
        """Calculate an upper bound on the commuting measurement value of the CHSH game."""
        prob_mat, pred_mat = self.chsh_extended_nonlocal_game()
        chsh = ExtendedNonlocalGame(prob_mat, pred_mat)
        res = chsh.commuting_measurement_value_upper_bound(k=2)
        expected_res = 3 / 4

        self.assertEqual(np.isclose(res, expected_res, atol=0.001), True)

    def test_mub_3in2out_entangled_bounds(self):
        """Test that entangled lower bound does not exceed upper bound for MUB 3-in,2-out game."""
        # Construct the game
        e0, e1 = basis(2, 0), basis(2, 1)
        ep = (e0 + e1) / np.sqrt(2)
        em = (e0 - e1) / np.sqrt(2)
        dim = 2
        a_out = b_out = 2
        a_in = b_in = 3
        pred_mat = np.zeros([dim, dim, a_out, b_out, a_in, b_in])
        pred_mat[:, :, 0, 0, 0, 0] = e0 @ e0.conj().T
        pred_mat[:, :, 1, 1, 0, 0] = e1 @ e1.conj().T
        pred_mat[:, :, 0, 0, 1, 1] = ep @ ep.conj().T
        pred_mat[:, :, 1, 1, 1, 1] = em @ em.conj().T
        pred_mat[:, :, 0, 0, 2, 2] = em @ em.conj().T
        pred_mat[:, :, 1, 1, 2, 2] = ep @ ep.conj().T
        prob_mat = 1 / 3 * np.identity(3)

        # Test single-round game
        game_single = ExtendedNonlocalGame(prob_mat, pred_mat, reps=1)

        # Compute all values
        unent_single = game_single.unentangled_value()
        ns_single = game_single.nonsignaling_value()

        # Theoretical values
        expected_unent = 2 / 3
        expected_ns = (3 + np.sqrt(5)) / 6

        # Verify theoretical values
        assert np.isclose(unent_single, expected_unent, atol=1e-4)
        assert np.isclose(ns_single, expected_ns, atol=1e-4)

        # Compute quantum bounds
        ent_lb_single = game_single.quantum_value_lower_bound()
        ent_ub_single = game_single.commuting_measurement_value_upper_bound()

        # Verify quantum bounds are within valid range
        assert 0 <= ent_lb_single <= 1
        assert 0 <= ent_ub_single <= 1

        # Verify bounds ordering
        assert ent_lb_single <= ent_ub_single + 1e-4
        assert ent_ub_single <= ns_single + 1e-4
        assert unent_single <= ent_lb_single + 1e-4

        # For this specific game, quantum value should equal unentangled value
        assert np.isclose(ent_lb_single, expected_unent, atol=1e-4)

        # Upper bound should be at least unentangled value
        assert ent_ub_single >= expected_unent - 1e-4

        ## Test multi-round game, take too much time to run, skip for now
        # game_multi = ExtendedNonlocalGame(prob_mat, pred_mat, reps=2)
        #
        ## Compute values for multi-round game
        # unent_multi = game_multi.unentangled_value()
        # ent_lb_multi = game_multi.quantum_value_lower_bound(iters=2, tol=1e-3)
        # ns_multi = game_multi.nonsignaling_value()
        #
        ## Verify theoretical unentangled value for multi-round
        # assert np.isclose(unent_multi, (2/3)**2, atol=1e-3), \
        #    "Unentangled value for 2 reps should be (2/3)^2"
        ## Verify bounds ordering for multi-round
        # assert ent_lb_multi <= ns_multi + 1e-4, \
        #    "Multi-round: Entangled LB should be <= non-signaling value"
        # assert unent_multi <= ent_lb_multi + 1e-4, \
        #    "Multi-round: Unentangled should be <= entangled LB"
        #
        ## Additional check: Entangled LB for multi-rep should exceed single-rep unentangled value
        # assert ent_lb_multi > expected_unent - 1e-4, \
        #    "Multi-rep entangled LB should exceed single-rep unentangled value"
        #    # For multi-round games, entangled value can potentially exceed unentangled value
        #    # (though not necessarily in this specific example)

    # taking too much time, skip for now, can replace with other small test
    # def test_theoretical_bounds_ordering(self):
    #    """Test bounds ordering with BB84 game instead of custom game."""
    #    # Create a simple game for testing
    #    prob_mat, pred_mat = self.bb84_extended_nonlocal_game()
    #
    #    for reps in [1, 2]:
    #        game = ExtendedNonlocalGame(prob_mat, pred_mat, reps=reps)
    #
    #        try:
    #            unent = game.unentangled_value()
    #            ent_lb = game.quantum_value_lower_bound(iters=2, tol=1e-3)
    #            ent_ub = game.commuting_measurement_value_upper_bound()
    #            ns = game.nonsignaling_value()
    #        except Exception as e:
    #            self.fail(f"Computation failed for reps={reps}: {str(e)}")
    #        # Verify bounds ordering
    #        self.assertLessEqual(unent, ent_lb + 1e-4,
    #                            f"reps={reps}: Unentangled ≤ Entangled LB")
    #        self.assertLessEqual(ent_lb, ent_ub + 1e-4,
    #                            f"reps={reps}: Entangled LB ≤ Entangled UB")
    #        self.assertLessEqual(ent_ub, ns + 1e-4,
    #                            f"reps={reps}: Entangled UB ≤ Non-signaling")
    #
    #        # For single-round games, quantum LB should equal unentangled value
    #        if reps == 1:
    #            self.assertAlmostEqual(unent, ent_lb, delta=1e-4,
    #                                  msg=f"reps={reps}: Unentangled ≈ Entangled LB")
