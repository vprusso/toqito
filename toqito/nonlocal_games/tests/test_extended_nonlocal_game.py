"""Tests for ExtendedNonlocalGame class."""

import unittest
from unittest import mock

import cvxpy
import numpy as np
import pytest

from toqito.nonlocal_games.extended_nonlocal_game import ExtendedNonlocalGame
from toqito.nonlocal_games.tests._constrained_games_fixtures import (
    bb84_extended_nonlocal_game,
    constrained_bb84_monogamy_answer_constraints,
    constrained_bb84_monogamy_answer_constraints_dense,
    forbid_bb84_diagonal_answers_at,
)
from toqito.rand import random_povm
from toqito.states import basis


class TestExtendedNonlocalGame(unittest.TestCase):
    """Unit test for ExtendedNonlocalGame."""

    @staticmethod
    def bb84_extended_nonlocal_game():
        """Define the BB84 extended nonlocal game."""
        return bb84_extended_nonlocal_game()

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

    def test_invalid_game_inputs(self):
        """The constructor rejects malformed predicates and improper question distributions."""
        prob_mat, pred_mat = self.bb84_extended_nonlocal_game()
        with pytest.raises(ValueError, match="must have 6 dimensions"):
            ExtendedNonlocalGame(prob_mat, pred_mat[0])
        with pytest.raises(ValueError, match="must be square"):
            ExtendedNonlocalGame(prob_mat, pred_mat[:1])
        with pytest.raises(ValueError, match="question pairs"):
            ExtendedNonlocalGame(np.full((3, 3), 1 / 9), pred_mat)
        with pytest.raises(ValueError, match="sum to 1"):
            ExtendedNonlocalGame(np.full((2, 2), 1 / 2), pred_mat)

    def test_game_dimensions(self):
        """The game records the referee dimension and the sizes of the question and answer sets."""
        prob_mat, pred_mat = self.moe_mub_4_in_3_out_game()
        moe = ExtendedNonlocalGame(prob_mat, pred_mat)
        self.assertEqual(
            (moe.referee_dim, moe.num_alice_out, moe.num_bob_out, moe.num_alice_in, moe.num_bob_in), (3, 3, 3, 4, 4)
        )

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
        """The see-saw attains the quantum value of the BB84 game, which equals its unentangled value."""
        prob_mat, pred_mat = self.bb84_extended_nonlocal_game()
        bb84 = ExtendedNonlocalGame(prob_mat, pred_mat)
        res = bb84.quantum_value_lower_bound(seed=0)
        self.assertAlmostEqual(res, np.cos(np.pi / 8) ** 2, delta=1e-4)

    def test_quantum_value_lower_bound_seed_reproducible(self):
        """A fixed seed makes the see-saw lower bound reproducible."""
        prob_mat, pred_mat = self.bb84_extended_nonlocal_game()
        bb84 = ExtendedNonlocalGame(prob_mat, pred_mat)
        res_1 = bb84.quantum_value_lower_bound(iters=2, seed=7)
        res_2 = bb84.quantum_value_lower_bound(iters=2, seed=7)
        self.assertEqual(res_1, res_2)

    def test_quantum_value_lower_bound_bob_dimension(self):
        """Bob's local dimension can exceed his number of outputs."""
        prob_mat, pred_mat = self.bb84_extended_nonlocal_game()
        bb84 = ExtendedNonlocalGame(prob_mat, pred_mat)
        res = bb84.quantum_value_lower_bound(dim=3, iters=2, seed=0)
        self.assertLessEqual(res, np.cos(np.pi / 8) ** 2 + 1e-4)
        self.assertGreater(res, 1 / 2)

    def test_quantum_value_lower_bound_initial_bob_povms(self):
        """Supplied starting measurements are used for the first restart and must have the right shape."""
        prob_mat, pred_mat = self.bb84_extended_nonlocal_game()
        bb84 = ExtendedNonlocalGame(prob_mat, pred_mat)
        start = random_povm(2, bb84.num_bob_in, bb84.num_bob_out, seed=3)

        res = bb84.quantum_value_lower_bound(iters=1, seed=0, initial_bob_povms=start)
        self.assertAlmostEqual(res, np.cos(np.pi / 8) ** 2, delta=1e-4)

        with pytest.raises(ValueError, match="initial_bob_povms"):
            bb84.quantum_value_lower_bound(iters=1, initial_bob_povms=start[:, :, :1, :])

    def test_quantum_value_lower_bound_step_cap(self):
        """With a tolerance that is never met, each restart runs `max_steps` steps and still returns a bound."""
        prob_mat, pred_mat = self.bb84_extended_nonlocal_game()
        bb84 = ExtendedNonlocalGame(prob_mat, pred_mat)
        res = bb84.quantum_value_lower_bound(iters=1, tol=0.0, seed=0, max_steps=2)
        self.assertLessEqual(res, np.cos(np.pi / 8) ** 2 + 1e-4)
        self.assertGreater(res, 1 / 2)

    def test_quantum_value_lower_bound_raises_when_solver_fails(self):
        """A see-saw step whose SDP does not solve is reported rather than returning a placeholder."""
        prob_mat, pred_mat = self.bb84_extended_nonlocal_game()
        bb84 = ExtendedNonlocalGame(prob_mat, pred_mat)
        with mock.patch.object(cvxpy.Problem, "solve", lambda self, *args, **kwargs: None):
            with pytest.raises(ValueError, match="did not solve successfully"):
                bb84.quantum_value_lower_bound(iters=1)

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

    @pytest.mark.slow
    def test_bb84_commuting_value_upper_bound(self):
        """Calculate an upper bound on the commuting measurement value of the BB84 game."""
        prob_mat, pred_mat = self.bb84_extended_nonlocal_game()
        bb84 = ExtendedNonlocalGame(prob_mat, pred_mat)
        res = bb84.commuting_measurement_value_upper_bound()
        expected_res = np.cos(np.pi / 8) ** 2

        self.assertEqual(np.isclose(res, expected_res, atol=1e-5), True)

    @pytest.mark.slow
    def test_constrained_bb84_commuting_value_upper_bound(self):
        """Constrained BB84 upper bound matches arXiv:2405.13717 Sec. 4.1, Eq. (62)."""
        prob_mat, pred_mat = self.bb84_extended_nonlocal_game()
        bb84 = ExtendedNonlocalGame(prob_mat, pred_mat)
        constraints = constrained_bb84_monogamy_answer_constraints()
        res = bb84.commuting_measurement_value_upper_bound(constraints=constraints)
        # omega_admiss^{(k=1)}(G^{BB84}_C) = cos^2(pi/8) from the paper.
        expected_res = 0.8535533905

        self.assertTrue(np.isclose(res, expected_res, atol=1e-5))

    @pytest.mark.slow
    def test_constrained_bb84_matches_unconstrained_sanity_check(self):
        """Forcing consistent answers does not lower the BB84 NPA upper bound at k=1."""
        prob_mat, pred_mat = self.bb84_extended_nonlocal_game()
        bb84 = ExtendedNonlocalGame(prob_mat, pred_mat)
        unconstrained = bb84.commuting_measurement_value_upper_bound()
        constrained = bb84.commuting_measurement_value_upper_bound(
            constraints=constrained_bb84_monogamy_answer_constraints()
        )

        self.assertTrue(np.isclose(unconstrained, constrained, atol=1e-5))
        self.assertTrue(np.isclose(unconstrained, np.cos(np.pi / 8) ** 2, atol=1e-5))

    def test_constrained_bb84_monogamy_constraints_are_per_question(self):
        """Eq. (60) is encoded as one equality per question x, not a single merged sum."""
        constraints = constrained_bb84_monogamy_answer_constraints()

        self.assertEqual(len(constraints), 2)
        self.assertEqual(set(constraints[0][0].keys()), {(0, 1, 0, 0), (1, 0, 0, 0)})
        self.assertEqual(set(constraints[1][0].keys()), {(0, 1, 1, 1), (1, 0, 1, 1)})
        self.assertEqual(constraints[0][1:], ("==", 0.0))
        self.assertEqual(constraints[1][1:], ("==", 0.0))

    def test_answer_event_constraint_invalid_operator(self):
        """Invalid comparison operators are rejected."""
        prob_mat, pred_mat = self.bb84_extended_nonlocal_game()
        bb84 = ExtendedNonlocalGame(prob_mat, pred_mat)
        bad_constraint = ({(0, 1, 0, 0): 1.0}, "!=", 0.0)  # type: ignore[misc]

        with self.assertRaises(ValueError):
            bb84.commuting_measurement_value_upper_bound(constraints=[bad_constraint])

    @pytest.mark.slow
    def test_bb84_binding_answer_event_constraint_lowers_commuting_upper_bound(self):
        """A binding constraint lowers the SDP value below the unconstrained BB84 upper bound."""
        prob_mat, pred_mat = self.bb84_extended_nonlocal_game()
        bb84 = ExtendedNonlocalGame(prob_mat, pred_mat)
        unconstrained = bb84.commuting_measurement_value_upper_bound()
        binding = bb84.commuting_measurement_value_upper_bound(constraints=forbid_bb84_diagonal_answers_at(0, 0))
        bb84_optimum = np.cos(np.pi / 8) ** 2

        self.assertTrue(np.isclose(unconstrained, bb84_optimum, atol=1e-5))
        self.assertLess(binding, bb84_optimum - 1e-4)
        self.assertLess(binding, unconstrained - 1e-4)
        # With both diagonal answer events at (x,y)=(0,0) forced to zero, only the (1,1) round
        # contributes at weight 1/2 in the BB84 distribution.
        self.assertAlmostEqual(binding, 0.5, delta=1e-4)

    @pytest.mark.slow
    def test_bb84_monogamy_constraints_dense_matches_sparse(self):
        """Dense coefficient arrays give the same upper bound as sparse dicts."""
        prob_mat, pred_mat = self.bb84_extended_nonlocal_game()
        bb84 = ExtendedNonlocalGame(prob_mat, pred_mat)
        sparse_val = bb84.commuting_measurement_value_upper_bound(
            constraints=constrained_bb84_monogamy_answer_constraints()
        )
        dense_val = bb84.commuting_measurement_value_upper_bound(
            constraints=constrained_bb84_monogamy_answer_constraints_dense()
        )

        self.assertTrue(np.isclose(sparse_val, dense_val, atol=1e-5))
        self.assertTrue(np.isclose(sparse_val, 0.8535533905, atol=1e-5))

    def test_answer_event_constraint_dense_shape_mismatch(self):
        """Dense coefficients with the wrong shape are rejected."""
        prob_mat, pred_mat = self.bb84_extended_nonlocal_game()
        bb84 = ExtendedNonlocalGame(prob_mat, pred_mat)
        bad_dense = (np.zeros((2, 2, 2, 3)), "==", 0.0)

        with self.assertRaises(ValueError):
            bb84.commuting_measurement_value_upper_bound(constraints=[bad_dense])

    @pytest.mark.slow
    def test_constrained_bb84_value_strictly_lower_than_unconstrained(self):
        """A binding answer-event constraint must lower the SDP value.

        forbid_bb84_diagonal_answers_at(0, 0) forces both p(0,0|0,0)=0 and
        p(1,1|0,0)=0, the only two winning answer pairs at (x=0,y=0).
        Since prob_mat[0,0]=0.5, this wipes out half the game's weight and
        the upper bound drops to about 0.5. Any regression that silently drops the
        constraint list will leave the bound at cos²(π/8) ≈ 0.854 and fail.
        """
        prob_mat, pred_mat = bb84_extended_nonlocal_game()
        game = ExtendedNonlocalGame(prob_mat, pred_mat)

        ub_unconstrained = game.commuting_measurement_value_upper_bound(k=1)
        bb84_optimum = np.cos(np.pi / 8) ** 2

        # forbid_bb84_diagonal_answers_at is already proven binding in
        # test_bb84_binding_answer_event_constraint_lowers_commuting_upper_bound.
        ub_constrained = game.commuting_measurement_value_upper_bound(
            k=1, constraints=forbid_bb84_diagonal_answers_at(0, 0)
        )

        self.assertTrue(np.isclose(ub_unconstrained, bb84_optimum, atol=1e-5))
        self.assertLess(
            ub_constrained,
            bb84_optimum - 1e-4,
            msg=(
                f"Constrained value {ub_constrained:.6f} should be strictly below "
                f"unconstrained value {ub_unconstrained:.6f}. "
                "The constraint machinery may be silently dropped."
            ),
        )
        # Both winning diagonal events at (x=0,y=0) are forbidden; only the
        # (x=1,y=1) round (weight 0.5) can contribute, so the bound is about 0.5.
        self.assertAlmostEqual(ub_constrained, 0.5, delta=1e-4)

    @pytest.mark.slow
    def test_constrained_bb84_dense_ndarray_coeffs_matches_sparse(self):
        """Dense np.ndarray coefficients must produce the same result as sparse dicts.

        Uses the same binding constraint as forbid_bb84_diagonal_answers_at(0, 0):
        p(0,0|0,0) + p(1,1|0,0) == 0. The dense array indexing is [a, b, x, y],
        so the two nonzero entries are coeffs[0,0,0,0] and coeffs[1,1,0,0]
        (not [1,1,1,1], which would be question pair (x=1,y=1)).
        Both paths must agree and the result must be strictly below cos²(π/8).
        """
        prob_mat, pred_mat = bb84_extended_nonlocal_game()
        game = ExtendedNonlocalGame(prob_mat, pred_mat)

        # Sparse dict: forbid both diagonal winners at (x=0, y=0).
        sparse_constraint = ({(0, 0, 0, 0): 1.0, (1, 1, 0, 0): 1.0}, "==", 0.0)
        ub_sparse = game.commuting_measurement_value_upper_bound(k=1, constraints=[sparse_constraint])

        # Dense ndarray: same constraint, shape (A_out=2, B_out=2, A_in=2, B_in=2).
        # Index order is [a, b, x, y].
        coeffs_dense = np.zeros((2, 2, 2, 2))
        coeffs_dense[0, 0, 0, 0] = 1.0  # a=0, b=0, x=0, y=0
        coeffs_dense[1, 1, 0, 0] = 1.0  # a=1, b=1, x=0, y=0
        dense_constraint = (coeffs_dense, "==", 0.0)
        ub_dense = game.commuting_measurement_value_upper_bound(k=1, constraints=[dense_constraint])

        self.assertAlmostEqual(
            ub_sparse,
            ub_dense,
            delta=1e-4,
            msg=(
                "Dense and sparse constraint paths produced different SDP values. "
                "The ndarray branch in _answer_event_linear_constraint may be broken."
            ),
        )
        # Both should be binding: constraint wipes out (x=0,y=0) and bound drops to about 0.5.
        self.assertLess(ub_sparse, np.cos(np.pi / 8) ** 2 - 1e-4)
        self.assertAlmostEqual(ub_sparse, 0.5, delta=1e-4)

    def test_chsh_commuting_value_upper_bound(self):
        """Calculate an upper bound on the commuting measurement value of the CHSH game."""
        prob_mat, pred_mat = self.chsh_extended_nonlocal_game()
        chsh = ExtendedNonlocalGame(prob_mat, pred_mat)
        res = chsh.commuting_measurement_value_upper_bound(k=2)
        expected_res = 3 / 4

        self.assertEqual(np.isclose(res, expected_res, atol=0.001), True)

    @staticmethod
    def moe_mub_3in2out_game_definition():
        """MUB 3-in, 2-out extended nonlocal game."""
        e0, e1 = basis(2, 0), basis(2, 1)
        ep = (e0 + e1) / np.sqrt(2)
        em = (e0 - e1) / np.sqrt(2)
        dim = 2
        a_out = b_out = 2
        a_in = b_in = 3
        pred_mat = np.zeros([dim, dim, a_out, b_out, a_in, b_in])

        # Define predicate matrices
        pred_mat[:, :, 0, 0, 0, 0] = e0 @ e0.conj().T
        pred_mat[:, :, 1, 1, 0, 0] = e1 @ e1.conj().T
        pred_mat[:, :, 0, 0, 1, 1] = ep @ ep.conj().T
        pred_mat[:, :, 1, 1, 1, 1] = em @ em.conj().T
        pred_mat[:, :, 0, 0, 2, 2] = em @ em.conj().T
        pred_mat[:, :, 1, 1, 2, 2] = ep @ ep.conj().T

        # Uniform probability distribution
        prob_mat = 1 / 3 * np.identity(3)

        return prob_mat, pred_mat

    def test_mub_3in2out_entangled_bounds_single_round(self):
        """Test bounds for the MUB 3-in, 2-out extended nonlocal game.

        Verifies individual bounds and their relationships. For this specific game:
        - Unentangled value is (3+sqrt(5))/6 (achieved by non-constant strategy f=(0,0,1)).
        - Quantum value equals the unentangled value (no quantum advantage).
        - NPA hierarchy level k=1 upper bound matches the unentangled value.
        - Non-signaling value is (3+sqrt(5))/6.
        """
        prob_mat_local, pred_mat_local = self.moe_mub_3in2out_game_definition()
        game = ExtendedNonlocalGame(prob_mat_local, pred_mat_local, reps=1)

        unent = game.unentangled_value()
        ns = game.nonsignaling_value()
        ent_lb = game.quantum_value_lower_bound(iters=1, tol=1e-7, seed=42)
        ent_ub = game.commuting_measurement_value_upper_bound(k=1)

        expected_value = (3 + np.sqrt(5)) / 6.0

        # 1. Verify individual known values
        self.assertAlmostEqual(unent, expected_value, delta=1e-4)
        self.assertAlmostEqual(ns, expected_value, delta=1e-4)
        self.assertAlmostEqual(ent_ub, expected_value, delta=5e-3)

        # 2. Verify universal ordering that MUST hold for valid bounds
        self.assertLessEqual(ent_lb, ent_ub + 1e-5)
        self.assertLessEqual(ent_ub, ns + 1e-5)
        self.assertLessEqual(ent_lb, ns + 1e-5)
        self.assertLessEqual(unent, ent_ub + 1e-5)

    @staticmethod
    def four_mub_game():
        """Define the 4-MUB extended nonlocal game (single round, qutrit bases)."""
        d = 3
        e0, e1, e2 = np.eye(3, dtype=complex)
        zeta = np.exp(2j * np.pi / 3)
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
        pi = np.zeros((num_inputs, num_inputs), dtype=float)
        for x in range(num_inputs):
            pi[x, x] = 1.0 / num_inputs
        pred_mat = np.zeros((d, d, num_outputs, num_outputs, num_inputs, num_inputs), dtype=complex)
        for x in range(num_inputs):
            for a in range(num_outputs):
                ket = B[x][a]
                pred_mat[:, :, a, a, x, x] = np.outer(ket, ket.conj())
        return pi, pred_mat

    def test_four_mub_unentangled_value(self):
        """Unentangled (classical) value of the 4-MUB game."""
        pi, pred_mat = self.four_mub_game()
        game = ExtendedNonlocalGame(pi, pred_mat)
        res = game.unentangled_value()
        expected = (3 + np.sqrt(5)) / 8
        self.assertAlmostEqual(res, expected, places=5)

    @pytest.mark.slow
    def test_four_mub_quantum_lower_bound(self):
        """Quantum heuristic lower bound of the 4-MUB game."""
        pi, pred_mat = self.four_mub_game()
        game = ExtendedNonlocalGame(pi, pred_mat)
        lb = game.quantum_value_lower_bound(seed=42, iters=3)
        # Entanglement beats the unentangled value (3 + sqrt(5)) / 8 of this game.
        self.assertAlmostEqual(lb, 0.660986, delta=1e-4)
        self.assertGreater(lb, (3 + np.sqrt(5)) / 8 + 1e-3)

    @pytest.mark.slow
    def test_four_mub_npa_upper_bound_k1ab(self):
        """NPA upper bound at k='1+ab' for the 4-MUB game."""
        pi, pred_mat = self.four_mub_game()
        game = ExtendedNonlocalGame(pi, pred_mat)
        ub = game.commuting_measurement_value_upper_bound(k=1, no_signaling=False)
        # With block-hermitian K (correct formulation), the bound is looser than
        # the old globally-hermitian formulation which gave ~0.761.
        self.assertAlmostEqual(ub, 0.8067, delta=5e-3)

    def test_four_mub_nonsignaling_value(self):
        """No-signaling value of the 4-MUB game."""
        pi, pred_mat = self.four_mub_game()
        game = ExtendedNonlocalGame(pi, pred_mat)
        ns = game.nonsignaling_value()
        self.assertAlmostEqual(ns, 0.788675, places=5)


class TestAnswerEventLinearConstraint(unittest.TestCase):
    """Unit tests for the answer-event constraint builder.

    These call the builder directly rather than going through
    ``commuting_measurement_value_upper_bound`` so that the operator and
    index-validation branches are exercised without paying for an SDP solve.
    """

    @staticmethod
    def _assemblage(referee_dim=2, num_a_out=2, num_b_out=2, num_a_in=2, num_b_in=2):
        """Build an assemblage of the shape the builder expects."""
        return {
            (x, y): cvxpy.Variable((num_a_out * referee_dim, num_b_out * referee_dim), hermitian=True)
            for x in range(num_a_in)
            for y in range(num_b_in)
        }

    def test_sparse_index_out_of_range(self):
        """A sparse key outside the game dimensions is rejected."""
        assemblage = self._assemblage()
        # num_b_out is 2, so b = 5 is out of range.
        constraint = ({(0, 5, 0, 0): 1.0}, "==", 0.0)

        with self.assertRaises(ValueError):
            ExtendedNonlocalGame._answer_event_linear_constraint(assemblage, constraint, 2, 2, 2, 2, 2)

    def test_less_than_or_equal_operator(self):
        """The '<=' operator produces an inequality constraint."""
        assemblage = self._assemblage()
        constraint = ({(0, 1, 0, 0): 1.0}, "<=", 0.5)

        built = ExtendedNonlocalGame._answer_event_linear_constraint(assemblage, constraint, 2, 2, 2, 2, 2)

        self.assertIsInstance(built, cvxpy.constraints.constraint.Constraint)

    def test_greater_than_or_equal_operator(self):
        """The '>=' operator produces an inequality constraint."""
        assemblage = self._assemblage()
        constraint = ({(0, 1, 0, 0): 1.0}, ">=", 0.25)

        built = ExtendedNonlocalGame._answer_event_linear_constraint(assemblage, constraint, 2, 2, 2, 2, 2)

        self.assertIsInstance(built, cvxpy.constraints.constraint.Constraint)


class TestExtendedNonlocalGameStrategies(unittest.TestCase):
    """The strategies returned alongside each value reproduce that value."""

    def test_unentangled_strategy_attains_value(self):
        """The optimal answers and referee state win with the unentangled value."""
        bb84 = ExtendedNonlocalGame(*bb84_extended_nonlocal_game())
        value, (alice_outputs, bob_outputs, referee_state) = bb84.unentangled_value(return_strategy=True)
        self.assertAlmostEqual(value, np.cos(np.pi / 8) ** 2)
        self.assertEqual(referee_state.shape, (2, 2))
        self.assertAlmostEqual(np.trace(referee_state).real, 1.0)
        self.assertAlmostEqual(bb84.unentangled_strategy_value(alice_outputs, bob_outputs, referee_state), value)

    def test_quantum_strategy_attains_lower_bound(self):
        """The returned state and measurements form a valid strategy whose winning probability is the bound."""
        bb84 = ExtendedNonlocalGame(*bb84_extended_nonlocal_game())
        value, strategy = bb84.quantum_value_lower_bound(seed=0, return_strategy=True)
        self.assertAlmostEqual(value, np.cos(np.pi / 8) ** 2, delta=1e-4)
        self.assertAlmostEqual(bb84.quantum_strategy_value(strategy), value, delta=1e-6)

        # Referee (2), Alice (referee_dim * dim = 4), Bob (2).
        self.assertEqual(strategy.state.shape, (16, 16))
        self.assertAlmostEqual(np.trace(strategy.state).real, 1.0, delta=1e-8)
        self.assertEqual(strategy.alice_povms.shape, (4, 4, 2, 2))
        self.assertEqual(strategy.bob_povms.shape, (2, 2, 2, 2))
        np.testing.assert_allclose(
            strategy.alice_povms.sum(axis=3), np.broadcast_to(np.eye(4)[:, :, np.newaxis], (4, 4, 2)), atol=1e-6
        )
        np.testing.assert_allclose(
            strategy.bob_povms.sum(axis=3), np.broadcast_to(np.eye(2)[:, :, np.newaxis], (2, 2, 2)), atol=1e-6
        )

    def test_nonsignaling_assemblage_attains_value(self):
        """The optimal assemblage is positive, normalised and wins with the non-signaling value."""
        bb84 = ExtendedNonlocalGame(*bb84_extended_nonlocal_game())
        value, assemblage = bb84.nonsignaling_value(return_strategy=True)
        self.assertAlmostEqual(bb84.assemblage_value(assemblage), value, delta=1e-6)
        self.assertEqual(assemblage.shape, (2, 2, 2, 2, 2, 2))
        referee_states = assemblage.sum(axis=(2, 3))
        for x in range(2):
            for y in range(2):
                np.testing.assert_allclose(referee_states[:, :, x, y], referee_states[:, :, 0, 0], atol=1e-5)
        self.assertAlmostEqual(np.trace(referee_states[:, :, 0, 0]).real, 1.0, delta=1e-5)

    def test_commuting_assemblage_attains_upper_bound(self):
        """The assemblage of the level-1 relaxation wins with the upper bound."""
        bb84 = ExtendedNonlocalGame(*bb84_extended_nonlocal_game())
        value, assemblage = bb84.commuting_measurement_value_upper_bound(k=1, return_strategy=True)
        self.assertAlmostEqual(bb84.assemblage_value(assemblage), value, delta=1e-6)
        self.assertEqual(assemblage.shape, (2, 2, 2, 2, 2, 2))
        self.assertAlmostEqual(np.trace(assemblage.sum(axis=(2, 3))[:, :, 0, 0]).real, 1.0, delta=1e-5)
