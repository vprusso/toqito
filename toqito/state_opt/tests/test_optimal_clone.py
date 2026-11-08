"""Tests for optimal_clone."""

import numpy as np
import pytest

from toqito.state_opt import optimal_clone
from toqito.states import basis

e_0, e_1 = basis(2, 0), basis(2, 1)
e_p = (e_0 + e_1) / np.sqrt(2)
e_m = (e_0 - e_1) / np.sqrt(2)

states = [e_0, e_1, e_p, e_m]
probs = [1 / 4, 1 / 4, 1 / 4, 1 / 4]


@pytest.mark.parametrize(
    "input_states, input_probs, num_reps, input_strategy, expected",
    [
        # Probability of counterfeit attack on Wiesner's quantum money
        (states, probs, 1, False, 3 / 4),
        # Probability of counterfeit attack on Wiesner's quantum money with 2 parallel repetitions
        (states, probs, 2, False, 3 / 4),
        # Counterfeit attack on Wiesner's quantum money (primal problem)
        (states, probs, 1, True, 3 / 4),
        # Counterfeit attack on Wiesner's quantum money (primal problem) with 2 parallel repetitions
        (states, probs, 2, True, 3 / 4),
    ],
)
def test_optimal_clone(input_states, input_probs, num_reps, input_strategy, expected):
    """Test functions work as expected."""
    expected_result = expected**num_reps
    calculated_result = optimal_clone(input_states, input_probs, num_reps, input_strategy)
    assert pytest.approx(expected_result, 1e-2) == calculated_result


@pytest.mark.parametrize(
    "input_states, input_probs, expected",
    [
        # Probability of counterfeit attack on Wiesner's quantum money
        (states, probs, 3 / 4),
    ],
)
def test_optimal_clone_default_reps_strategy(input_states, input_probs, expected):
    """Test functions work as expected."""
    expected_result = expected
    calculated_result = optimal_clone(input_states, input_probs)
    assert pytest.approx(expected_result, 1e-2) == calculated_result


e_y_p = (e_0 + 1j * e_1) / np.sqrt(2)
e_y_m = (e_0 - 1j * e_1) / np.sqrt(2)
six_states = [e_0, e_1, e_p, e_m, e_y_p, e_y_m]


@pytest.mark.parametrize("num_reps, expected", [(1, 2 / 3), (2, 4 / 9)])
@pytest.mark.parametrize("strategy", [False, True])
def test_optimal_clone_six_state_ensemble(num_reps, strategy, expected):
    """The six-state ensemble has complex amplitudes; the dual must stay Hermitian to reach the primal value."""
    assert optimal_clone(six_states, [1 / 6] * 6, num_reps, strategy) == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_optimal_clone_dual_matches_primal_complex_states(seed):
    """Strong duality holds for random complex qubit ensembles."""
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(3, 2, 1)) + 1j * rng.normal(size=(3, 2, 1))
    random_states = [vec / np.linalg.norm(vec) for vec in vectors]
    random_probs = [1 / 3] * 3
    primal = optimal_clone(random_states, random_probs, 1, True)
    dual = optimal_clone(random_states, random_probs, 1, False)
    assert dual == pytest.approx(primal, abs=1e-4)
