"""Test symmetric_extension_hierarchy."""
import numpy as np
import pytest

from toqito.perms import swap
from toqito.state_opt import symmetric_extension_hierarchy
from toqito.states import basis, bell


def test_symmetric_extension_hierarchy_four_bell_density_matrices():
    """Symmetric extension hierarchy for four Bell density matrices."""
    states = [
        bell(0) * bell(0).conj().T,
        bell(1) * bell(1).conj().T,
        bell(2) * bell(2).conj().T,
        bell(3) * bell(3).conj().T,
    ]
    res = symmetric_extension_hierarchy(states=states, probs=None, level=2)
    np.testing.assert_equal(np.isclose(res, 1 / 2), True)


def test_symmetric_extension_hierarchy_four_bell_states():
    """Symmetric extension hierarchy for four Bell states."""
    states = [bell(0), bell(1), bell(2), bell(3)]
    res = symmetric_extension_hierarchy(states=states, probs=None, level=2)
    np.testing.assert_equal(np.isclose(res, 1 / 2), True)


def test_symmetric_extension_hierarchy_four_bell_with_resource_state_lvl_1():
    """Level 1 of hierarchy for four Bell states and resource state."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    e_00, e_11 = np.kron(e_0, e_0), np.kron(e_1, e_1)

    eps = 1 / 2
    eps_state = np.sqrt((1 + eps) / 2) * e_00 + np.sqrt((1 - eps) / 2) * e_11
    eps_dm = eps_state * eps_state.conj().T

    states = [
        np.kron(bell(0) * bell(0).conj().T, eps_dm),
        np.kron(bell(1) * bell(1).conj().T, eps_dm),
        np.kron(bell(2) * bell(2).conj().T, eps_dm),
        np.kron(bell(3) * bell(3).conj().T, eps_dm),
    ]

    # Ensure we are checking the correct partition of the states.
    states = [
        swap(states[0], [2, 3], [2, 2, 2, 2]),
        swap(states[1], [2, 3], [2, 2, 2, 2]),
        swap(states[2], [2, 3], [2, 2, 2, 2]),
        swap(states[3], [2, 3], [2, 2, 2, 2]),
    ]

    # Level 1 of the hierarchy should be identical to the known PPT value
    # for this case.
    res = symmetric_extension_hierarchy(states=states, probs=None, level=1)
    exp_res = 1 / 2 * (1 + np.sqrt(1 - eps ** 2))

    np.testing.assert_equal(np.isclose(res, exp_res), True)


@pytest.mark.skip(reason="This test takes too much time.")  # pylint: disable=not-callable
def test_symmetric_extension_hierarchy_four_bell_with_resource_state():
    """Symmetric extension hierarchy for four Bell states and resource state."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    e_00, e_11 = np.kron(e_0, e_0), np.kron(e_1, e_1)

    eps = 1 / 2
    eps_state = np.sqrt((1 + eps) / 2) * e_00 + np.sqrt((1 - eps) / 2) * e_11
    eps_dm = eps_state * eps_state.conj().T

    states = [
        np.kron(bell(0) * bell(0).conj().T, eps_dm),
        np.kron(bell(1) * bell(1).conj().T, eps_dm),
        np.kron(bell(2) * bell(2).conj().T, eps_dm),
        np.kron(bell(3) * bell(3).conj().T, eps_dm),
    ]

    # Ensure we are checking the correct partition of the states.
    states = [
        swap(states[0], [2, 3], [2, 2, 2, 2]),
        swap(states[1], [2, 3], [2, 2, 2, 2]),
        swap(states[2], [2, 3], [2, 2, 2, 2]),
        swap(states[3], [2, 3], [2, 2, 2, 2]),
    ]

    res = symmetric_extension_hierarchy(states=states, probs=None, level=2)
    exp_res = 1 / 2 * (1 + np.sqrt(1 - eps ** 2))

    np.testing.assert_equal(np.isclose(res, exp_res), True)


@pytest.mark.skip(reason="This test takes too much time.")  # pylint: disable=not-callable
def test_symmetric_extension_hierarchy_three_bell_with_resource_state():
    """Symmetric extension hierarchy for three Bell and resource state."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    e_00, e_11 = np.kron(e_0, e_0), np.kron(e_1, e_1)

    eps = 1 / 2
    eps_state = np.sqrt((1 + eps) / 2) * e_00 + np.sqrt((1 - eps) / 2) * e_11
    eps_dm = eps_state * eps_state.conj().T

    states = [
        np.kron(bell(0) * bell(0).conj().T, eps_dm),
        np.kron(bell(1) * bell(1).conj().T, eps_dm),
        np.kron(bell(2) * bell(2).conj().T, eps_dm),
    ]

    # Ensure we are checking the correct partition of the states.
    states = [
        swap(states[0], [2, 3], [2, 2, 2, 2]),
        swap(states[1], [2, 3], [2, 2, 2, 2]),
        swap(states[2], [2, 3], [2, 2, 2, 2]),
    ]

    res = symmetric_extension_hierarchy(states=states, probs=None, level=2)
    np.testing.assert_equal(np.isclose(res, 0.9583057), True)


def test_invalid_symmetric_extension_hierarchy_probs():
    """Invalid probability vector for symmetric extension hierarchy."""
    with np.testing.assert_raises(ValueError):
        rho1 = bell(0) * bell(0).conj().T
        rho2 = bell(1) * bell(1).conj().T
        states = [rho1, rho2]
        symmetric_extension_hierarchy(states, [1, 2, 3])


def test_invalid_symmetric_extension_hierarchy_states():
    """Invalid number of states for symmetric_extension_hierarchy."""
    with np.testing.assert_raises(ValueError):
        states = []
        symmetric_extension_hierarchy(states)


if __name__ == "__main__":
    np.testing.run_module_suite()
