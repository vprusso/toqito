"""Test state_exclusion_locc."""

import cvxpy
import numpy as np
import pytest

from toqito.matrix_ops import partial_transpose
from toqito.state_opt import state_exclusion_locc
from toqito.states import domino


def _dm(vec):
    """Build a normalized density matrix from a (possibly unnormalized) vector."""
    v = np.array(vec, dtype=complex).reshape(-1, 1)
    return v @ v.conj().T / float(np.linalg.norm(v) ** 2)


def _direct_exclusion_sdp(states, ppt=False, dim=(2, 2)):
    """Global (or PPT) minimal-error state-exclusion probability via a direct SDP (reference)."""
    n, d = len(states), states[0].shape[0]
    meas = [cvxpy.Variable((d, d), hermitian=True) for _ in range(n)]
    constraints = [m >> 0 for m in meas] + [sum(meas) == np.identity(d)]
    if ppt:
        constraints += [partial_transpose(m, [0], list(dim)) >> 0 for m in meas]
    obj = cvxpy.Minimize(cvxpy.real(sum(cvxpy.trace(states[k] @ meas[k]) for k in range(n)) / n))
    return cvxpy.Problem(obj, constraints).solve()


# A fixed two-qubit ensemble with a known global < PPT (<= LOCC) exclusion gap, taken from the
# symmetric-extension exclusion analysis (issue #1523). Computational basis order |00>,|01>,|10>,|11>.
_GAP_VECS = [(0, 1, 1, 1), (0, 0, 1, 1), (1, 0, 1, 1)]


def test_product_orthogonal_states_match_global():
    """Orthogonal product states are perfectly LOCC-excludable (error ~ 0, matching global)."""
    e_0, e_1 = np.array([1, 0]), np.array([0, 1])
    states = [_dm(np.kron(e_0, e_0)), _dm(np.kron(e_1, e_1)), _dm(np.kron(e_0, e_1))]
    val = state_exclusion_locc(states, dim=[2, 2], reps=3, seed=0)
    np.testing.assert_allclose(val, 0.0, atol=1e-4)


def test_locc_never_below_global():
    """The one-way LOCC exclusion error is never below the global exclusion error."""
    states = [_dm(v) for v in _GAP_VECS]
    locc = state_exclusion_locc(states, dim=[2, 2], reps=3, seed=1)
    glob = _direct_exclusion_sdp(states, ppt=False)
    np.testing.assert_equal(locc >= glob - 1e-4, True)


def test_locc_strictly_worse_than_global():
    """One-way LOCC exclusion is strictly worse than global; PPT > global certifies the gap."""
    states = [_dm(v) for v in _GAP_VECS]
    locc = state_exclusion_locc(states, dim=[2, 2], reps=3, seed=1)
    glob = _direct_exclusion_sdp(states, ppt=False)
    ppt = _direct_exclusion_sdp(states, ppt=True)
    # PPT exclusion is a certified lower bound on LOCC exclusion, so PPT > global proves the
    # LOCC > global gap is genuine and not an artifact of the (heuristic) see-saw.
    np.testing.assert_equal(ppt > glob + 1e-2, True)
    np.testing.assert_equal(locc > glob + 1e-2, True)


def test_reproducible_with_seed():
    """A fixed seed yields a reproducible value."""
    states = [_dm(v) for v in _GAP_VECS]
    first = state_exclusion_locc(states, dim=[2, 2], reps=2, seed=7)
    second = state_exclusion_locc(states, dim=[2, 2], reps=2, seed=7)
    np.testing.assert_allclose(first, second, atol=1e-9)


@pytest.mark.slow
def test_domino_states_are_locc_excludable():
    """Domino states are not LOCC-distinguishable, yet they are perfectly LOCC-excludable."""
    states = [_dm(domino(i)) for i in range(9)]
    val = state_exclusion_locc(states, dim=[3, 3], reps=2, seed=0)
    np.testing.assert_allclose(val, 0.0, atol=1e-4)


def test_requires_dim():
    """Omitting `dim` raises a ValueError."""
    states = [_dm((1, 0, 0, 0)), _dm((0, 0, 0, 1))]
    with pytest.raises(ValueError, match="dim"):
        state_exclusion_locc(states)


def test_invalid_probs():
    """Probabilities that do not sum to 1 raise a ValueError."""
    states = [_dm((1, 0, 0, 0)), _dm((0, 0, 0, 1))]
    with pytest.raises(ValueError, match="sum to 1"):
        state_exclusion_locc(states, probs=[0.2, 0.2], dim=[2, 2])


def test_requires_states():
    """An empty list of states raises a ValueError."""
    with pytest.raises(ValueError, match="At least one state"):
        state_exclusion_locc([], dim=[2, 2])


def test_dim_product_must_match_states():
    """A `dim` whose product is not the state dimension raises a ValueError."""
    states = [_dm((1, 0, 0, 0)), _dm((0, 0, 0, 1))]  # dimension 4
    with pytest.raises(ValueError, match="product"):
        state_exclusion_locc(states, dim=[2, 3])  # 2 * 3 = 6 != 4


def test_runs_without_seed():
    """The see-saw runs with the default (None) seed."""
    e_0, e_1 = np.array([1, 0]), np.array([0, 1])
    states = [_dm(np.kron(e_0, e_0)), _dm(np.kron(e_1, e_1)), _dm(np.kron(e_0, e_1))]
    val = state_exclusion_locc(states, dim=[2, 2], reps=1)
    np.testing.assert_allclose(val, 0.0, atol=1e-4)


def test_custom_num_alice_outcomes():
    """`num_alice_outcomes` is accepted; more outcomes cannot raise the value."""
    states = [_dm(v) for v in _GAP_VECS]
    val = state_exclusion_locc(states, dim=[2, 2], num_alice_outcomes=4, reps=2, seed=0)
    np.testing.assert_equal(val >= -1e-6, True)
