"""Tests for optimal_clone."""
import pytest
import numpy as np

from toqito.states import basis
from toqito.state_opt import optimal_clone

e_0, e_1 = basis(2, 0), basis(2, 1)
e_p = (e_0 + e_1) / np.sqrt(2)
e_m = (e_0 - e_1) / np.sqrt(2)

states = [e_0, e_1, e_p, e_m]
probs = [1 / 4, 1 / 4, 1 / 4, 1 / 4]


def test_counterfeit_attack_wiesner_money():
    """Probability of counterfeit attack on Wiesner's quantum money."""
    res = optimal_clone(states, probs)
    np.testing.assert_equal(np.isclose(res, 3 / 4), True)


def test_counterfeit_attack_wiesner_money_rep_2():
    """Probability of counterfeit attack with 2 parallel repetitions."""
    reps = 2

    res = optimal_clone(states, probs, reps)
    np.testing.assert_equal(np.isclose(res, (3 / 4) ** reps), True)


def test_counterfeit_attack_wiesner_money_primal_problem():
    """Counterfeit attack on Wiesner's quantum money (primal problem)."""
    res = optimal_clone(states, probs, 1, True)
    np.testing.assert_equal(np.isclose(res, 3 / 4), True)


def test_counterfeit_attack_wiesner_money_primal_problem_rep_1():
    """Counterfeit attack with 1 parallel repetitions (primal problem)."""
    reps = 1
    res = optimal_clone(states, probs, reps, True)
    np.testing.assert_equal(np.isclose(res, (3 / 4)), True)


@pytest.mark.skip(reason="This test takes too much time.")  # pylint: disable=not-callable
def test_counterfeit_attack_wiesner_money_primal_problem_rep_2():
    """Counterfeit attack with 2 parallel repetitions (primal problem)."""
    reps = 2
    res = optimal_clone(states, probs, reps, True)
    np.testing.assert_equal(np.isclose(res, (3 / 4) ** reps), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
