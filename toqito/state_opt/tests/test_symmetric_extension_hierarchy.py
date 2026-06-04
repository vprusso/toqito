"""Test symmetric_extension_hierarchy."""

import cvxpy
import numpy as np
import pytest

from toqito.matrix_ops import partial_transpose
from toqito.perms import swap
from toqito.state_opt import symmetric_extension_hierarchy
from toqito.states import basis, bell, werner


def test_symmetric_extension_hierarchy_four_bell_density_matrices():
    """Symmetric extension hierarchy for four Bell density matrices."""
    states = [
        bell(0) @ bell(0).conj().T,
        bell(1) @ bell(1).conj().T,
        bell(2) @ bell(2).conj().T,
        bell(3) @ bell(3).conj().T,
    ]
    res = symmetric_extension_hierarchy(states=states, probs=None, level=2)
    np.testing.assert_equal(np.isclose(res, 1 / 2, atol=1e-5), True)


def test_symmetric_extension_hierarchy_four_bell_states():
    """Symmetric extension hierarchy for four Bell states."""
    states = [bell(0), bell(1), bell(2), bell(3)]
    res = symmetric_extension_hierarchy(states=states, probs=None, level=2)
    np.testing.assert_equal(np.isclose(res, 1 / 2, atol=1e-5), True)


def test_symmetric_extension_hierarchy_four_bell_with_resource_state_lvl_1():
    """Level 1 of hierarchy for four Bell states and resource state."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    e_00, e_11 = np.kron(e_0, e_0), np.kron(e_1, e_1)

    eps = 1 / 2
    eps_state = np.sqrt((1 + eps) / 2) * e_00 + np.sqrt((1 - eps) / 2) * e_11
    eps_dm = eps_state * eps_state.conj().T

    states = [
        np.kron(bell(0) @ bell(0).conj().T, eps_dm),
        np.kron(bell(1) @ bell(1).conj().T, eps_dm),
        np.kron(bell(2) @ bell(2).conj().T, eps_dm),
        np.kron(bell(3) @ bell(3).conj().T, eps_dm),
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
    exp_res = 1 / 2 * (1 + np.sqrt(1 - eps**2))

    np.testing.assert_equal(np.isclose(res, exp_res), True)


@pytest.mark.skip(reason="This test takes too much time.")
def test_symmetric_extension_hierarchy_four_bell_with_resource_state():
    """Symmetric extension hierarchy for four Bell states and resource state."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    e_00, e_11 = np.kron(e_0, e_0), np.kron(e_1, e_1)

    eps = 1 / 2
    eps_state = np.sqrt((1 + eps) / 2) * e_00 + np.sqrt((1 - eps) / 2) * e_11
    eps_dm = eps_state * eps_state.conj().T

    states = [
        np.kron(bell(0) @ bell(0).conj().T, eps_dm),
        np.kron(bell(1) @ bell(1).conj().T, eps_dm),
        np.kron(bell(2) @ bell(2).conj().T, eps_dm),
        np.kron(bell(3) @ bell(3).conj().T, eps_dm),
    ]

    # Ensure we are checking the correct partition of the states.
    states = [
        swap(states[0], [2, 3], [2, 2, 2, 2]),
        swap(states[1], [2, 3], [2, 2, 2, 2]),
        swap(states[2], [2, 3], [2, 2, 2, 2]),
        swap(states[3], [2, 3], [2, 2, 2, 2]),
    ]

    res = symmetric_extension_hierarchy(states=states, probs=None, level=2)
    exp_res = 1 / 2 * (1 + np.sqrt(1 - eps**2))

    np.testing.assert_equal(np.isclose(res, exp_res), True)


@pytest.mark.skip(reason="This test takes too much time.")
def test_symmetric_extension_hierarchy_three_bell_with_resource_state():
    """Symmetric extension hierarchy for three Bell and resource state."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    e_00, e_11 = np.kron(e_0, e_0), np.kron(e_1, e_1)

    eps = 1 / 2
    eps_state = np.sqrt((1 + eps) / 2) * e_00 + np.sqrt((1 - eps) / 2) * e_11
    eps_dm = eps_state @ eps_state.conj().T

    states = [
        np.kron(bell(0) @ bell(0).conj().T, eps_dm),
        np.kron(bell(1) @ bell(1).conj().T, eps_dm),
        np.kron(bell(2) @ bell(2).conj().T, eps_dm),
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
        rho1 = bell(0) @ bell(0).conj().T
        rho2 = bell(1) @ bell(1).conj().T
        states = [rho1, rho2]
        symmetric_extension_hierarchy(states, [1, 2, 3])


def test_invalid_symmetric_extension_hierarchy_states():
    """Invalid number of states for symmetric_extension_hierarchy."""
    with np.testing.assert_raises(ValueError):
        states = []
        symmetric_extension_hierarchy(states)


def test_invalid_symmetric_extension_hierarchy_states_none():
    """None as states raises ValueError."""
    with np.testing.assert_raises(ValueError):
        symmetric_extension_hierarchy(None)


def test_invalid_symmetric_extension_hierarchy_probs_not_summing_to_one():
    """Probabilities that do not sum to 1 raise ValueError."""
    with np.testing.assert_raises(ValueError):
        rho = bell(0) @ bell(0).conj().T
        symmetric_extension_hierarchy([rho], [0.5])


def test_symmetric_extension_hierarchy_extremal_werner_states():
    """Symmetric extension hierarchy for two extremal Werner states."""
    dim = 5
    states = [werner(dim, -1.0), werner(dim, 1.0)]

    # See: [Cos15] Cosentino, Alessandro.
    #     "Quantum State Local Distinguishability via Convex Optimization"
    # Section 4.3 An example: Werner hiding pair
    upper_bound = 0.5 + 1 / (dim + 1)
    res = symmetric_extension_hierarchy(states=states, probs=None, level=1)
    atol = 1e-5
    np.testing.assert_equal(res <= upper_bound + atol, True)


def test_symmetric_extension_hierarchy_scalar_dim_must_divide():
    """A scalar `dim` that does not evenly divide the state length should raise a clear error."""
    states = [bell(0) @ bell(0).conj().T]
    with pytest.raises(ValueError, match="evenly divide"):
        symmetric_extension_hierarchy(states=states, probs=[1.0], level=1, dim=3)


def _gap_ensemble():
    """Return a fixed two-qubit ensemble whose separable exclusion error exceeds the global one.

    The three (unnormalized) pure states, in the computational basis ordered as
    (|00>, |01>, |10>, |11>), are |01> + |10> + |11>, |10> + |11>, and |00> + |10> + |11>. They
    are not antidistinguishable, so the exclusion error is strictly positive, and restricting to
    PPT/separable measurements makes it strictly larger than the global exclusion error.
    """
    vecs = [
        np.array([[0], [1], [1], [1]], dtype=complex),
        np.array([[0], [0], [1], [1]], dtype=complex),
        np.array([[1], [0], [1], [1]], dtype=complex),
    ]
    return [v @ v.conj().T / (v.conj().T @ v).real.item() for v in vecs]


def _direct_exclusion_sdp(states, ppt=False, dim=(2, 2)):
    """Minimal-error state-exclusion probability via a direct SDP (reference value).

    With `ppt=True` the measurement operators are additionally constrained to be PPT, which
    yields the value the level-1 symmetric-extension hierarchy should reproduce.
    """
    n, d = len(states), states[0].shape[0]
    meas = [cvxpy.Variable((d, d), hermitian=True) for _ in range(n)]
    constraints = [m >> 0 for m in meas] + [sum(meas) == np.identity(d)]
    if ppt:
        constraints += [partial_transpose(m, [0], list(dim)) >> 0 for m in meas]
    obj = cvxpy.Minimize(cvxpy.real(sum(cvxpy.trace(states[k] @ meas[k]) for k in range(n)) / n))
    return cvxpy.Problem(obj, constraints).solve()


def test_exclusion_orthogonal_states_perfectly_excludable():
    """Orthogonal Bell states are perfectly antidistinguishable: the exclusion error is zero."""
    states = [bell(i) @ bell(i).conj().T for i in range(4)]
    res = symmetric_extension_hierarchy(states=states, probs=None, level=1, objective="exclude")
    np.testing.assert_allclose(res, 0.0, atol=1e-5)


def test_exclusion_level_one_matches_ppt_value():
    """Level 1 of the exclusion hierarchy equals the PPT state-exclusion value."""
    states = _gap_ensemble()
    res = symmetric_extension_hierarchy(states=states, probs=None, level=1, objective="exclude")
    np.testing.assert_allclose(res, _direct_exclusion_sdp(states, ppt=True), atol=1e-4)


def test_exclusion_separable_strictly_worse_than_global():
    """The separable (level-1/PPT) exclusion error is strictly larger than the global one."""
    states = _gap_ensemble()
    sep = symmetric_extension_hierarchy(states=states, probs=None, level=1, objective="exclude")
    glob = _direct_exclusion_sdp(states, ppt=False)
    np.testing.assert_equal(sep > glob + 1e-2, True)


def test_exclusion_hierarchy_is_monotonic():
    """Higher levels give a non-decreasing lower bound on the separable exclusion error."""
    states = _gap_ensemble()
    lvl_1 = symmetric_extension_hierarchy(states=states, probs=None, level=1, objective="exclude")
    lvl_2 = symmetric_extension_hierarchy(states=states, probs=None, level=2, objective="exclude")
    # For this ensemble the PPT relaxation is already tight, so level 2 coincides with level 1 to
    # solver precision; the test asserts only that the hierarchy never decreases with the level.
    np.testing.assert_equal(lvl_2 >= lvl_1 - 1e-5, True)


def test_invalid_objective_raises():
    """An unknown `objective` value raises a ValueError."""
    states = [bell(0) @ bell(0).conj().T, bell(1) @ bell(1).conj().T]
    with pytest.raises(ValueError, match="objective"):
        symmetric_extension_hierarchy(states=states, level=1, objective="antidistinguish")
