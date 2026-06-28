"""Test state_exclusion."""

import numpy as np
import pytest

from toqito.matrices import standard_basis
from toqito.matrix_ops import to_density_matrix
from toqito.perms import swap
from toqito.state_opt import state_exclusion, symmetric_extension_hierarchy
from toqito.states import basis, bell

e_0, e_1 = standard_basis(2)

states_min_error = [
    # Bell states are perfectly distinguishable, so the probability of error should be nil
    ([bell(0), bell(1), bell(2), bell(3)], None, 0, {}),
    ([bell(0), bell(1), bell(2), bell(3)], [1 / 4, 1 / 4, 1 / 4, 1 / 4], 0, {}),
    ([to_density_matrix(bell(0)), to_density_matrix(bell(1))], None, 0, {"cvxopt_kktsolver": "ldl"}),
    ([to_density_matrix(bell(0)), to_density_matrix(bell(1))], [1 / 2, 1 / 2], 0, {"cvxopt_kktsolver": "ldl"}),
    # For |0> and |+>, this probability is 1/2 - 1/(2*sqrt(2))
    (
        [np.array([[1.0], [0.0]]), np.array([[1.0], [1.0]]) / np.sqrt(2)],
        None,
        0.14644660940672627,
        {"cvxopt_kktsolver": "ldl"},
    ),
    (
        [np.array([[1.0], [0.0]]), np.array([[1.0], [1.0]]) / np.sqrt(2)],
        [1 / 2, 1 / 2],
        0.14644660940672627,
        {"cvxopt_kktsolver": "ldl"},
    ),
]

states_unambiguous = [
    # Bell states are perfectly distinguishable, so the probability of error should be nil
    ([bell(0), bell(1), bell(2), bell(3)], None, 0, {}),
    ([bell(0), bell(1), bell(2), bell(3)], [1 / 4, 1 / 4, 1 / 4, 1 / 4], 0, {}),
    ([to_density_matrix(bell(0)), to_density_matrix(bell(1))], None, 0, {}),
    ([to_density_matrix(bell(0)), to_density_matrix(bell(1))], [1 / 2, 1 / 2], 0, {}),
    # For |0> and |+>, this probability is 1/sqrt(2)
    (
        [np.array([[1.0], [0.0]]), np.array([[1.0], [1.0]]) / np.sqrt(2)],
        None,
        0.707106781186547,
        {
            "abs_ipm_opt_tol": 1e-5,
        },
    ),
    (
        [np.array([[1.0], [0.0]]), np.array([[1.0], [1.0]]) / np.sqrt(2)],
        [1 / 2, 1 / 2],
        0.707106781186547,
        {
            "abs_ipm_opt_tol": 1e-5,
        },
    ),
]

solvers = ["cvxopt"]

primal_duals = ["primal", "dual"]


def _dm(vec):
    """Build a normalized density matrix from a vector."""
    vector = np.array(vec, dtype=complex).reshape(-1, 1)
    return vector @ vector.conj().T / float(np.linalg.norm(vector) ** 2)


def _gap_ensemble():
    """Return a two-qubit ensemble with a strict global/PPT exclusion gap."""
    return [_dm(vec) for vec in [(0, 1, 1, 1), (0, 0, 1, 1), (1, 0, 1, 1)]]


def _bell_resource_ensemble():
    """Return four Bell states tensored with a partially entangled resource state."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    e_00, e_11 = np.kron(e_0, e_0), np.kron(e_1, e_1)

    eps = 1 / 2
    resource_state = np.sqrt((1 + eps) / 2) * e_00 + np.sqrt((1 - eps) / 2) * e_11
    resource_dm = resource_state @ resource_state.conj().T

    states = [np.kron(bell(idx) @ bell(idx).conj().T, resource_dm) for idx in range(4)]
    return [swap(state, [2, 3], [2, 2, 2, 2]) for state in states]


@pytest.mark.parametrize("vectors, probs, expected_result, kwargs", states_min_error)
@pytest.mark.parametrize("solver", solvers)
@pytest.mark.parametrize("primal_dual", primal_duals)
def test_state_exclusion_min_error(vectors, probs, solver, primal_dual, expected_result, kwargs):
    """Test function works as expected for a valid input."""
    val, _ = state_exclusion(vectors=vectors, probs=probs, solver=solver, primal_dual=primal_dual, **kwargs)
    assert abs(val - expected_result) <= 1e-8


@pytest.mark.parametrize("vectors, probs, expected_result, kwargs", states_unambiguous)
@pytest.mark.parametrize("solver", solvers)
@pytest.mark.parametrize("primal_dual", primal_duals)
def test_state_exclusion_unambiguous(vectors, probs, solver, primal_dual, expected_result, kwargs):
    """Test function works as expected for a valid input."""
    val, _ = state_exclusion(
        vectors=vectors, probs=probs, solver=solver, primal_dual=primal_dual, strategy="unambiguous", **kwargs
    )
    # Accuracy is quite low bcause of primals=None
    assert abs(val - expected_result) <= 1e-3


@pytest.mark.parametrize(
    "vectors, probs, solver, primal_dual",
    [
        # Bell states (default uniform probs with dual).
        ([bell(0), bell(1), bell(2), e_0], None, "cvxopt", "dual"),
    ],
)
@pytest.mark.parametrize(
    "strategy",
    [
        "min_error",
        "unambiguous",
    ],
)
def test_state_exclusion_invalid_vectors(vectors, probs, solver, primal_dual, strategy):
    """Test function works as expected for an invalid input."""
    with pytest.raises(ValueError, match="Vectors for state distinguishability must all have the same dimension."):
        state_exclusion(vectors=vectors, probs=probs, solver=solver, primal_dual=primal_dual, strategy=strategy)


@pytest.mark.parametrize("primal_dual", primal_duals)
@pytest.mark.parametrize("strategy", ["min_error", "unambiguous"])
def test_state_exclusion_ppt_four_bell_states(primal_dual, strategy):
    """PPT exclusion supports primal and dual SDPs for both exclusion strategies."""
    val, _ = state_exclusion(
        vectors=[bell(0), bell(1), bell(2), bell(3)],
        measurement="ppt",
        subsystems=[0],
        dimensions=[2, 2],
        strategy=strategy,
        primal_dual=primal_dual,
        cvxopt_kktsolver="ldl",
    )
    np.testing.assert_allclose(val, 0, atol=1e-6)


def test_state_exclusion_ppt_gap_ensemble_matches_symmetric_extension_level_one():
    """PPT exclusion agrees with level 1 of the symmetric-extension exclusion hierarchy."""
    states = _gap_ensemble()

    ppt_primal, _ = state_exclusion(
        states,
        measurement="ppt",
        subsystems=[0],
        dimensions=[2, 2],
        primal_dual="primal",
        cvxopt_kktsolver="ldl",
    )
    ppt_dual, _ = state_exclusion(
        states,
        measurement="ppt",
        subsystems=[0],
        dimensions=[2, 2],
        primal_dual="dual",
        cvxopt_kktsolver="ldl",
    )
    level_one = symmetric_extension_hierarchy(states=states, probs=None, level=1, objective="exclude")

    np.testing.assert_allclose(ppt_primal, ppt_dual, atol=1e-5)
    np.testing.assert_allclose(ppt_primal, level_one, atol=1e-4)


def test_state_exclusion_ppt_is_strictly_worse_than_global_for_gap_ensemble():
    """PPT exclusion is bounded below by global exclusion and can be strictly larger."""
    states = _gap_ensemble()

    global_val, _ = state_exclusion(states, primal_dual="primal", cvxopt_kktsolver="ldl")
    ppt_val, _ = state_exclusion(
        states,
        measurement="ppt",
        subsystems=[0],
        dimensions=[2, 2],
        primal_dual="primal",
        cvxopt_kktsolver="ldl",
    )

    np.testing.assert_equal(ppt_val >= global_val - 1e-5, True)
    np.testing.assert_equal(ppt_val > global_val + 1e-2, True)


def test_state_exclusion_ppt_bell_resource_matches_level_one_exclusion():
    """PPT exclusion of Bell states with a resource state matches the hierarchy reference value."""
    states = _bell_resource_ensemble()

    ppt_val, _ = state_exclusion(
        states,
        measurement="ppt",
        subsystems=[0, 2],
        dimensions=[2, 2, 2, 2],
        primal_dual="dual",
        cvxopt_kktsolver="ldl",
    )
    level_one = symmetric_extension_hierarchy(states=states, probs=None, level=1, objective="exclude")

    np.testing.assert_allclose(ppt_val, level_one, atol=1e-4)


def test_state_exclusion_ppt_requires_subsystems_and_dimensions():
    """Using `measurement='ppt'` without subsystem data should raise a clear ValueError."""
    with pytest.raises(ValueError, match="subsystems.*dimensions"):
        state_exclusion(vectors=[bell(0), bell(1)], measurement="ppt")


def test_state_exclusion_ppt_wrong_dimensions():
    """PPT dimensions must multiply to the state dimension."""
    with pytest.raises(ValueError, match="product of `dimensions`"):
        state_exclusion(
            vectors=[bell(0), bell(1)],
            measurement="ppt",
            subsystems=[0],
            dimensions=[2, 3],
        )


def test_state_exclusion_ppt_subsystems_out_of_range():
    """PPT subsystem indices must be valid for the provided dimensions."""
    with pytest.raises(ValueError, match="index into `dimensions`"):
        state_exclusion(
            vectors=[bell(0), bell(1)],
            measurement="ppt",
            subsystems=[5],
            dimensions=[2, 2],
        )


def test_state_exclusion_invalid_measurement():
    """Unsupported measurement types should raise a clear ValueError."""
    with pytest.raises(ValueError, match="measurement.*positive.*ppt"):
        state_exclusion(vectors=[bell(0), bell(1)], measurement="random_string")


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"probs": [0.5]}, "must equal the number of states"),
        ({"probs": [-0.5, 1.5]}, "nonnegative"),
        ({"strategy": "bogus"}, "strategy must be either"),
    ],
)
def test_state_exclusion_invalid_inputs(kwargs, match):
    """Invalid probs/strategy are rejected before the SDP is set up."""
    vectors = [bell(0), bell(1)]
    with pytest.raises(ValueError, match=match):
        state_exclusion(vectors, **kwargs)
