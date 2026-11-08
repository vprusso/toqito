"""Tests for PostSelectionGame."""

import itertools
from unittest import mock

import cvxpy
import numpy as np
import pytest

from toqito.nonlocal_games import NonlocalGame, PostSelectionGame, QuantumStrategy
from toqito.nonlocal_games.post_selection_game import _best_response_ratios
from toqito.rand import random_density_matrix, random_povm

HARDY_PROBABILITY = (5 * np.sqrt(5) - 11) / 2


def chsh_game():
    """Build the CHSH game as a nonlocal game and as a post-selection game that keeps every round."""
    prob_mat = np.full((2, 2), 1 / 4)
    pred_mat = np.zeros((2, 2, 2, 2))
    for a, b, x, y in itertools.product(range(2), repeat=4):
        pred_mat[a, b, x, y] = float(a ^ b == x * y)
    return NonlocalGame(prob_mat, pred_mat), PostSelectionGame(prob_mat, pred_mat, np.ones_like(pred_mat))


def non_possibilistic_game():
    """Build the game G_not of Calleja Rodríguez, Bocanegra-Garay and Araújo, which encodes no paradox."""
    select_mat = np.zeros((2, 2, 2, 2))
    pred_mat = np.zeros((2, 2, 2, 2))
    for a, b, x, y in [(0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0), (0, 0, 1, 1)]:
        select_mat[a, b, x, y] = 1
    pred_mat[0, 0, 0, 1] = pred_mat[0, 0, 1, 0] = 1
    return PostSelectionGame(np.full((2, 2), 1 / 4), pred_mat, select_mat)


def random_game(rng, alice_out, bob_out, alice_in, bob_in):
    """Draw a post-selection game with random question distribution and winning and keeping probabilities."""
    prob_mat = rng.random((alice_in, bob_in))
    shape = (alice_out, bob_out, alice_in, bob_in)
    return PostSelectionGame(prob_mat / prob_mat.sum(), rng.random(shape), rng.random(shape))


def brute_force_classical_value(game):
    """Compute the classical value by enumerating every pair of deterministic strategies."""
    alice_out, bob_out, alice_in, bob_in = game.pred_mat.shape
    best = -np.inf
    for alice in itertools.product(range(alice_out), repeat=alice_in):
        for bob in itertools.product(range(bob_out), repeat=bob_in):
            try:
                best = max(best, game.classical_strategy_value(np.array(alice), np.array(bob)))
            except ValueError:
                continue
    return best


def deterministic_distribution(game, alice_outputs, bob_outputs):
    """Return the conditional distribution of answers of a deterministic strategy."""
    distribution = np.zeros(game.pred_mat.shape)
    for x, a in enumerate(alice_outputs):
        for y, b in enumerate(bob_outputs):
            distribution[a, b, x, y] = 1
    return distribution


class TestHardyGame:
    """The generalized Hardy games of the paper."""

    @pytest.mark.parametrize("num_inputs, num_outputs", [(2, 2), (3, 2), (2, 3), (3, 3), (4, 2)])
    def test_classical_value_is_one_half(self, num_inputs, num_outputs):
        """Every generalized Hardy game has classical value 1/2, attained by a deterministic strategy."""
        game = PostSelectionGame.hardy(num_inputs, num_outputs)
        value, (alice_outputs, bob_outputs) = game.classical_value(return_strategy=True)
        assert value == pytest.approx(0.5)
        assert game.classical_strategy_value(alice_outputs, bob_outputs) == pytest.approx(0.5)

    def test_explicit_strategy_attains_classical_value(self):
        """Bob always answers 1 and Alice answers 1 except on her last question, where she answers 0.

        The only kept events are then the winning one at the last question pair and the losing one at questions
        (s-1, s-2).
        """
        for num_inputs in (2, 3, 4):
            game = PostSelectionGame.hardy(num_inputs, 2)
            alice_outputs = np.ones(num_inputs, dtype=int)
            alice_outputs[-1] = 0
            assert game.classical_strategy_value(alice_outputs, np.ones(num_inputs, dtype=int)) == pytest.approx(0.5)

    def test_question_distribution_is_uniform_over_the_paradox(self):
        """The 2s question pairs entering the paradox are asked with equal probability and nothing else is asked."""
        game = PostSelectionGame.hardy(3, 2)
        asked = game.prob_mat > 0
        assert asked.sum() == 6
        np.testing.assert_allclose(game.prob_mat[asked], 1 / 6)
        assert game.select_mat.sum(axis=(0, 1))[~asked].sum() == 0

    @pytest.mark.parametrize("num_inputs, num_outputs", [(2, 2), (3, 2), (2, 3)])
    def test_quantum_value_is_one(self, num_inputs, num_outputs):
        """The NPA bound reaches the quantum value 1 already at level 1+ab."""
        game = PostSelectionGame.hardy(num_inputs, num_outputs)
        assert game.commuting_measurement_value_upper_bound(k="1+ab") == pytest.approx(1.0, abs=1e-5)

    def test_nonsignaling_value_is_one(self):
        """A PR box satisfies every Hardy condition with a positive winning event."""
        game = PostSelectionGame.hardy()
        value, distribution = game.nonsignaling_value(return_strategy=True)
        assert value == pytest.approx(1.0, abs=1e-7)
        assert game.distribution_value(distribution) == pytest.approx(1.0, abs=1e-6)
        np.testing.assert_allclose(distribution.sum(axis=(0, 1)), 1, atol=1e-6)

    def test_maximal_hardy_probability(self):
        """The largest post-selection probability of a behaviour of value one is the Hardy probability over 4."""
        game = PostSelectionGame.hardy()
        kept, distribution = game.post_selection_probability_upper_bound(1.0, k="1+ab", return_strategy=True)
        assert kept == pytest.approx(HARDY_PROBABILITY / 4, abs=1e-6)
        assert game.post_selection_probability(distribution) == pytest.approx(kept, abs=1e-8)
        assert game.distribution_value(distribution) == pytest.approx(1.0, abs=1e-5)
        # The paper's statistical power for the optimal Hardy behaviour, (5 sqrt(5) - 11) / 8 ~ 0.0225.
        assert game.statistical_power(distribution) == pytest.approx(HARDY_PROBABILITY / 4, abs=1e-4)

    def test_rejects_degenerate_sizes(self):
        """The paradox needs two questions and two answers."""
        with pytest.raises(ValueError, match="at least two"):
            PostSelectionGame.hardy(1, 2)
        with pytest.raises(ValueError, match="at least two"):
            PostSelectionGame.hardy(2, 1)


class TestNonPossibilisticGame:
    """The paper's game G_not, whose quantum value is only reached as the post-selection probability vanishes."""

    def test_bounds(self):
        """Classical value 1/2 and quantum and no-signaling value 2/3."""
        game = non_possibilistic_game()
        assert game.classical_value() == pytest.approx(0.5)
        assert game.nonsignaling_value() == pytest.approx(2 / 3, abs=1e-6)
        assert game.commuting_measurement_value_upper_bound(k=1) == pytest.approx(2 / 3, abs=1e-5)
        assert game.commuting_measurement_value_upper_bound(k=2) == pytest.approx(2 / 3, abs=1e-5)

    def test_maximal_statistical_power(self):
        """At value 0.5931 the largest post-selection probability is about 0.2451, for a power of about 0.00617."""
        game = non_possibilistic_game()
        kept, distribution = game.post_selection_probability_upper_bound(0.5931, k=2, return_strategy=True)
        assert kept == pytest.approx(0.2451, abs=5e-4)
        assert game.distribution_value(distribution) >= 0.5931 - 1e-6
        assert game.statistical_power(distribution) == pytest.approx(0.00617, abs=5e-5)

    def test_unreachable_value_has_zero_post_selection_probability(self):
        """Above the quantum value only behaviours that are never kept satisfy the constraint."""
        assert non_possibilistic_game().post_selection_probability_upper_bound(0.9, k=1) == pytest.approx(0, abs=1e-7)


class TestAgreesWithNonlocalGame:
    """Keeping every round turns a post-selection game into the nonlocal game with the same predicate."""

    def test_chsh_values(self):
        """Classical value 3/4, Tsirelson bound cos^2(pi/8) and no-signaling value 1."""
        nonlocal_game, game = chsh_game()
        assert game.classical_value() == pytest.approx(nonlocal_game.classical_value())
        assert game.classical_value() == pytest.approx(0.75)
        assert game.nonsignaling_value() == pytest.approx(1.0, abs=1e-7)
        bound, distribution = game.commuting_measurement_value_upper_bound(k=1, return_strategy=True)
        assert bound == pytest.approx(np.cos(np.pi / 8) ** 2, abs=1e-6)
        assert game.post_selection_probability(distribution) == pytest.approx(1.0, abs=1e-6)
        assert nonlocal_game.distribution_value(distribution) == pytest.approx(bound, abs=1e-6)
        # The paper's statistical power of the ideal CHSH behaviour, D((2 + sqrt 2) / 4 || 3 / 4) ~ 0.0463.
        assert game.statistical_power(distribution) == pytest.approx(0.0463, abs=1e-4)

    def test_random_games_classical_value(self):
        """Random nonlocal games give the same classical value through either class."""
        rng = np.random.default_rng(7)
        for alice_out, bob_out, alice_in, bob_in in [(2, 2, 2, 2), (3, 2, 2, 3), (2, 3, 3, 2)]:
            prob_mat = rng.random((alice_in, bob_in))
            prob_mat /= prob_mat.sum()
            pred_mat = rng.random((alice_out, bob_out, alice_in, bob_in))
            nonlocal_game = NonlocalGame(prob_mat, pred_mat)
            game = PostSelectionGame(prob_mat, pred_mat, np.ones_like(pred_mat))
            assert game.classical_value() == pytest.approx(nonlocal_game.classical_value())

    def test_quantum_strategy_value(self):
        """A quantum strategy's value matches the nonlocal game's when every round is kept."""
        nonlocal_game, game = chsh_game()
        strategy = QuantumStrategy(
            state=random_density_matrix(4, seed=3),
            alice_povms=random_povm(2, 2, 2, seed=4),
            bob_povms=random_povm(2, 2, 2, seed=5),
        )
        assert game.quantum_strategy_value(strategy) == pytest.approx(nonlocal_game.quantum_strategy_value(strategy))


class TestClassicalValue:
    """The Dinkelbach enumeration against brute force."""

    @pytest.mark.parametrize("shape", [(2, 2, 2, 2), (3, 2, 2, 3), (2, 3, 3, 2), (2, 2, 3, 3), (3, 3, 2, 2)])
    def test_matches_brute_force(self, shape):
        """Random games with generic real-valued winning and keeping probabilities."""
        rng = np.random.default_rng(sum(shape))
        for _ in range(3):
            game = random_game(rng, *shape)
            value, (alice_outputs, bob_outputs) = game.classical_value(return_strategy=True)
            assert value == pytest.approx(brute_force_classical_value(game), abs=1e-10)
            assert game.classical_strategy_value(alice_outputs, bob_outputs) == pytest.approx(value, abs=1e-12)
            distribution = deterministic_distribution(game, alice_outputs, bob_outputs)
            assert game.distribution_value(distribution) == pytest.approx(value, abs=1e-12)

    def test_sparse_games_match_brute_force(self):
        """Games where most events are discarded, so many strategies never pass post-selection."""
        rng = np.random.default_rng(11)
        for _ in range(5):
            shape = (2, 3, 3, 2)
            prob_mat = np.full((3, 2), 1 / 6)
            select_mat = (rng.random(shape) < 0.2).astype(float)
            pred_mat = (rng.random(shape) < 0.5).astype(float)
            if not select_mat.any():
                continue
            game = PostSelectionGame(prob_mat, pred_mat, select_mat)
            assert game.classical_value() == pytest.approx(brute_force_classical_value(game), abs=1e-12)

    def test_chunked_enumeration(self):
        """Enumerating one strategy at a time gives the same value."""
        game = random_game(np.random.default_rng(5), 3, 2, 3, 2)
        expected, strategy = game.classical_value(return_strategy=True)
        with mock.patch("toqito.nonlocal_games.post_selection_game._STRATEGY_CHUNK_BYTES", 1):
            value, chunked_strategy = game.classical_value(return_strategy=True)
        assert value == pytest.approx(expected)
        assert game.classical_strategy_value(*chunked_strategy) == pytest.approx(expected)

    def test_enumerates_the_smaller_player(self):
        """The optimal strategy is returned in Alice-then-Bob order whichever player was enumerated."""
        rng = np.random.default_rng(2)
        for shape in [(2, 3, 2, 3), (3, 2, 3, 2)]:
            game = random_game(rng, *shape)
            value, (alice_outputs, bob_outputs) = game.classical_value(return_strategy=True)
            assert alice_outputs.shape == (shape[2],)
            assert bob_outputs.shape == (shape[3],)
            assert game.classical_strategy_value(alice_outputs, bob_outputs) == pytest.approx(value)

    def test_stops_early_at_value_one(self):
        """A game won on every kept event has value one, found without visiting every strategy."""
        pred_mat = np.ones((2, 2, 2, 2))
        select_mat = np.zeros((2, 2, 2, 2))
        select_mat[0, 0, 0, 0] = 1
        game = PostSelectionGame(np.full((2, 2), 1 / 4), pred_mat, select_mat)
        assert game.classical_value() == 1.0

    def test_best_response_ratios_without_wins(self):
        """When nothing is ever won the ratio is zero, with answers that still pass post-selection."""
        wins = np.zeros((1, 2, 2))
        keeps = np.array([[[0.0, 1.0], [0.0, 0.0]]])
        ratios, responses = _best_response_ratios(wins, keeps)
        assert ratios[0] == 0.0
        assert np.take_along_axis(keeps[0], responses[0][np.newaxis, :], axis=0).sum() > 0

    def test_best_response_ratios_without_post_selection(self):
        """A row whose every answer is discarded has ratio minus infinity."""
        ratios, _ = _best_response_ratios(np.zeros((1, 2, 2)), np.zeros((1, 2, 2)))
        assert ratios[0] == -np.inf


class TestValidation:
    """Rejected inputs and undefined values."""

    def test_shape_mismatch(self):
        """The selection probabilities must have the predicate's shape."""
        with pytest.raises(ValueError, match="select_mat"):
            PostSelectionGame(np.full((2, 2), 1 / 4), np.ones((2, 2, 2, 2)), np.ones((2, 2, 2, 3)))

    def test_entries_outside_unit_interval(self):
        """Winning and keeping probabilities lie in [0, 1]."""
        with pytest.raises(ValueError, match="pred_mat"):
            PostSelectionGame(np.full((2, 2), 1 / 4), 2 * np.ones((2, 2, 2, 2)), np.ones((2, 2, 2, 2)))
        with pytest.raises(ValueError, match="select_mat"):
            PostSelectionGame(np.full((2, 2), 1 / 4), np.ones((2, 2, 2, 2)), -np.ones((2, 2, 2, 2)))

    def test_never_kept(self):
        """Some asked event must be kept."""
        select_mat = np.zeros((2, 2, 2, 2))
        select_mat[:, :, 1, 1] = 1
        prob_mat = np.zeros((2, 2))
        prob_mat[0, 0] = 1
        with pytest.raises(ValueError, match="ever kept"):
            PostSelectionGame(prob_mat, np.ones((2, 2, 2, 2)), select_mat)

    def test_undefined_values(self):
        """Strategies and distributions that never pass post-selection have no value."""
        game = PostSelectionGame.hardy()
        with pytest.raises(ValueError, match="undefined"):
            game.classical_strategy_value(np.array([1, 1]), np.array([1, 1]))
        with pytest.raises(ValueError, match="undefined"):
            game.distribution_value(deterministic_distribution(game, [1, 1], [1, 1]))
        assert game.post_selection_probability(deterministic_distribution(game, [1, 1], [1, 1])) == 0.0

    def test_scs_defaults(self):
        """SCS gets a tight tolerance by default and may be given explicit parameters."""
        game = PostSelectionGame.hardy()
        assert game.commuting_measurement_value_upper_bound(k=1, solver=cvxpy.SCS) == pytest.approx(1.0, abs=1e-4)
        value = game.commuting_measurement_value_upper_bound(k=1, solver=cvxpy.SCS, solver_params={"eps": 1e-6})
        assert value == pytest.approx(1.0, abs=1e-3)
