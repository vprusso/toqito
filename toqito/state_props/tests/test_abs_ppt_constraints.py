"""Test abs_ppt_constraints."""

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

def test_qubit_qudit_constraints():
    """Test qubit-qudit constraints matrices."""
    eigs = np.random.rand(2 * 100)
    cons = abs_ppt_constraints(eigs, 2)
    expected_cons = analytical_qubit_qudit_constraints(eigs)
    assert cons[0] == pytest.approx(expected_cons[0])

def test_qutrit_qudit_constraints():
    """Test qutrit-qudit constraints matrices."""
    eigs = np.random.rand(3 * 100)
    cons = abs_ppt_constraints(eigs, 3)
    expected_cons = analytical_qutrit_qudit_constraints(eigs)
    assert (cons[0] == pytest.approx(expected_cons[0]) and cons[1] == pytest.approx(expected_cons[1])
            or cons[1] == pytest.approx(expected_cons[0]) and cons[0] == pytest.approx(expected_cons[1]))

def test_lim_cons():
    """Test that lim_cons correctly limits number of constraints."""
    eigs = np.random.rand(7 * 100)
    assert len(abs_ppt_constraints(eigs, 7, lim_cons=4000)) == 4000
