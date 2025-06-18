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
    "eigs, p, analytical_constraints",
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


@pytest.mark.parametrize(
    "upto, use_check, expected_counts",
    [
        # use_check == False
        (6, False, [0, 1, 2, 12, 286, 33592]),
        # use_check == True
        (5, True, [0, 1, 2, 10, 114]),
    ],
)
def test_constraint_counts(upto, use_check, expected_counts):
    """Test that the constraint counts match those in the docstring."""
    constraint_counts = []
    for n in range(1, upto + 1):
        eigs = np.random.rand(n * n)
        constraint_counts.append(len(abs_ppt_constraints(eigs, n, use_check=use_check)))
    assert constraint_counts == expected_counts


@pytest.mark.parametrize(
    "n",
    [
        2,
        5,
    ],
)
def test_cvxpy_case(n):
    """Test that the function does not throw an error when passed a cvxpy Variable.

    A separate test for checking if the output is correct for cvxpy Variables is not necessary since the algorithm uses
    the same algorithm in both cases.
    """
    eigs = cvxpy.Variable(n * n, nonneg=True)
    abs_ppt_constraints(eigs, n)


@pytest.mark.parametrize(
    "eigs, argslist, expected",
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


@pytest.mark.parametrize(
    "mat, dim, error_msg",
    [
        # Invalid input type
        ([1, 2, 3, 4], 2, "mat must be a numpy ndarray or a cvxpy Variable"),
        # Invalid input type
        ([cvxpy.Variable(1), cvxpy.Variable(1)], 2, "mat must be a numpy ndarray or a cvxpy Variable"),
    ],
)
def test_invalid(mat, dim, error_msg):
    """Test error-checking."""
    with pytest.raises(TypeError, match=error_msg):
        abs_ppt_constraints(mat, dim)
