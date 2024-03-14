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
        # Probability of counterfeit attack on Wiesner's quantum money with 2 parallel repitions
        (states, probs, 2, False, 3 / 4),
        # Counterfeit attack on Wiesner's quantum money (primal problem)
        (states, probs, 1, True, 3 / 4),
        # Counterfeit attack on Wiesner's quantum money (primal problem) with 2 parallel repitions
        (states, probs, 2, True, 3 / 4),
    ],
)
def test_optimal_clone(input_states, input_probs, num_reps, input_strategy, expected):
    """Test functions work as expected."""
    expected_result = expected**num_reps
    calculated_result = optimal_clone(input_states, input_probs, num_reps, input_strategy)
    assert pytest.approx(expected_result, 0.1) == calculated_result


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
    assert pytest.approx(expected_result, 0.1) == calculated_result
