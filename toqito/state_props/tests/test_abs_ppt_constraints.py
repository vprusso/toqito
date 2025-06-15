"""Test abs_ppt_constraints."""

from itertools import permutations

import cvxpy
import numpy as np
import pytest

from toqito.state_props import abs_ppt_constraints


def analytical_qubit_qudit_constraints(eigs):
    """Analytical constraint matrices for 2 x n bipartite state :cite:`Hildebrand_2007_AbsPPT`."""
    eigs = np.sort(eigs)[::-1]
    cons1 = np.array([[eigs[-1], eigs[-2]], [-eigs[0], eigs[-3]]])
    return [cons1 + cons1.T]


def analytical_qutrit_qudit_constraints(eigs):
    """Analytical constraint matrices for 3 x n bipartite state :cite:`Hildebrand_2007_AbsPPT`."""
    eigs = np.sort(eigs)[::-1]
    cons1 = np.array([[eigs[-1], eigs[-2], eigs[-4]], [-eigs[0], eigs[-3], eigs[-5]], [-eigs[1], -eigs[2], eigs[-6]]])
    cons2 = np.array([[eigs[-1], eigs[-2], eigs[-3]], [-eigs[0], eigs[-4], eigs[-5]], [-eigs[1], -eigs[2], eigs[-6]]])
    return [cons1 + cons1.T, cons2 + cons2.T]


@pytest.mark.parametrize(
    "eigs,p,analytical_constraints",
    [
        # Qubit-qudit constraints
        (np.random.rand(2 * 100), 2, analytical_qubit_qudit_constraints),
        # Qutrit-qudit constraints
        (np.random.rand(3 * 100), 3, analytical_qutrit_qudit_constraints),
    ],
)
def test_constraints(eigs, p, analytical_constraints):
    """Test constraint matrices."""
    cons = abs_ppt_constraints(eigs, p)
    expected_cons = analytical_constraints(eigs)
    any_match = False
    for cons_ordered in permutations(cons):
        all_match = True
        for computed_cons, analytical_cons in zip(cons_ordered, expected_cons):
            all_match &= computed_cons == pytest.approx(analytical_cons)
        any_match |= all_match
    assert any_match


def test_cvxpy_case():
    """Test that the function does not throw an error when passed a cvxpy Variable.

    A separate test for checking if the output is correct for cvxpy Variables is not necessary since the algorithm uses
    the same algorithm in both cases.
    """
    eigs = cvxpy.Variable(5 * 5, nonneg=True)
    abs_ppt_constraints(eigs, 5)


@pytest.mark.parametrize(
    "eigs,argslist,expected",
    [
        # Test that max_constraints correctly limits number of constraints
        (np.random.rand(7 * 100), [7, 4000], 4000),
        # Test that an empty list is returned when p = 1
        (np.random.rand(1 * 100), [1], 0),
    ],
)
def test_limiting_cases(eigs, argslist, expected):
    """Test various limiting cases."""
    assert len(abs_ppt_constraints(eigs, *argslist)) == expected
