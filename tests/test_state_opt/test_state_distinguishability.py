"""Test state_distinguishability."""
import numpy as np

from toqito.state_opt import state_distinguishability
from toqito.states import basis, bell


def test_state_distinguishability_one_state():
    """State distinguishability for single state."""
    rho = bell(0) * bell(0).conj().T
    states = [rho]

    res = state_distinguishability(states)
    np.testing.assert_equal(np.isclose(res, 1), True)


def test_state_distinguishability_one_state_vec():
    """State distinguishability for single vector state."""
    rho = bell(0)
    states = [rho]

    res = state_distinguishability(states)
    np.testing.assert_equal(np.isclose(res, 1), True)


def test_state_distinguishability_two_states():
    """State distinguishability for two state density matrices."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    e_00 = e_0 * e_0.conj().T
    e_11 = e_1 * e_1.conj().T
    states = [e_00, e_11]
    probs = [1 / 2, 1 / 2]

    res = state_distinguishability(states, probs)
    np.testing.assert_equal(np.isclose(res, 1), True)


def test_unambiguous_state_distinguishability_two_states():
    """Unambiguous state distinguishability for two state density matrices."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    e_00 = e_0 * e_0.conj().T
    e_11 = e_1 * e_1.conj().T
    states = [e_00, e_11]
    probs = [1 / 2, 1 / 2]

    res = state_distinguishability(states, probs, dist_method="unambiguous")
    np.testing.assert_equal(np.isclose(res, 1), True)


def test_state_distinguishability_three_state_vec():
    """State distinguishability for two state vectors."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    states = [e_0, e_1]
    probs = [1 / 2, 1 / 2]

    res = state_distinguishability(states, probs)
    np.testing.assert_equal(np.isclose(res, 1), True)


def test_state_distinguishability_yyd_density_matrices():
    """Global distinguishability of the YYD states should yield 1."""
    psi0 = bell(0) * bell(0).conj().T
    psi1 = bell(1) * bell(1).conj().T
    psi2 = bell(2) * bell(2).conj().T
    psi3 = bell(3) * bell(3).conj().T

    states = [
        np.kron(psi0, psi0),
        np.kron(psi2, psi1),
        np.kron(psi3, psi1),
        np.kron(psi1, psi1),
    ]
    probs = [1 / 4, 1 / 4, 1 / 4, 1 / 4]

    res = state_distinguishability(states, probs)
    np.testing.assert_equal(np.isclose(res, 1, atol=0.001), True)


def test_invalid_state_distinguishability_probs():
    """Invalid probability vector for state distinguishability."""
    with np.testing.assert_raises(ValueError):
        rho1 = bell(0) * bell(0).conj().T
        rho2 = bell(1) * bell(1).conj().T
        states = [rho1, rho2]
        state_distinguishability(states, [1, 2, 3])


def test_invalid_state_distinguishability_states():
    """Invalid number of states for state distinguishability."""
    with np.testing.assert_raises(ValueError):
        states = []
        state_distinguishability(states)


if __name__ == "__main__":
    np.testing.run_module_suite()
