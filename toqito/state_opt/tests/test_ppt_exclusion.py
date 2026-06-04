"""Test PPT state exclusion via state_exclusion."""

import numpy as np
import pytest

from toqito.state_opt import state_exclusion
from toqito.states import bell


def _random_bipartite_pure_states() -> list[np.ndarray]:
    """Generate a deterministic 2-by-2 pure-state ensemble with nontrivial PPT exclusion."""
    rng = np.random.default_rng(1520)
    states = []
    for _ in range(3):
        state = rng.normal(size=(4, 1)) + 1j * rng.normal(size=(4, 1))
        states.append(state / np.linalg.norm(state))
    return states


def test_ppt_exclusion_min_error_primal_dual_is_nontrivial():
    """PPT-constrained exclusion can be strictly worse than global exclusion."""
    states = _random_bipartite_pure_states()

    global_primal, _ = state_exclusion(
        states,
        primal_dual="primal",
        cvxopt_kktsolver="ldl",
    )
    global_dual, _ = state_exclusion(
        states,
        primal_dual="dual",
        cvxopt_kktsolver="ldl",
    )
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

    assert np.isclose(global_primal, global_dual, atol=1e-7)
    assert np.isclose(ppt_primal, ppt_dual, atol=1e-7)
    assert np.isclose(global_primal, 0.005152559655429941, atol=1e-7)
    assert np.isclose(ppt_primal, 0.0076139904965218975, atol=1e-7)
    assert ppt_primal > global_primal + 1e-3


def test_ppt_exclusion_unambiguous_primal_dual_bell_states():
    """Unambiguous PPT exclusion agrees between primal and dual on Bell states."""
    states = [bell(index) for index in range(4)]

    primal_res, _ = state_exclusion(
        states,
        measurement="ppt",
        subsystems=[0],
        dimensions=[2, 2],
        strategy="unambiguous",
        primal_dual="primal",
        cvxopt_kktsolver="ldl",
        abs_ipm_opt_tol=1e-7,
    )
    dual_res, _ = state_exclusion(
        states,
        measurement="ppt",
        subsystems=[0],
        dimensions=[2, 2],
        strategy="unambiguous",
        primal_dual="dual",
        cvxopt_kktsolver="ldl",
        abs_ipm_opt_tol=1e-7,
    )

    assert np.isclose(primal_res, 0, atol=1e-7)
    assert np.isclose(dual_res, 0, atol=1e-7)


def test_ppt_exclusion_unambiguous_primal_dual_nonorthogonal_qubits():
    """For one subsystem, PPT is equivalent to positivity and matches the positive-measurement value."""
    states = [np.array([[1.0], [0.0]]), np.array([[1.0], [1.0]]) / np.sqrt(2)]

    primal_res, _ = state_exclusion(
        states,
        measurement="ppt",
        subsystems=[0],
        dimensions=[2],
        strategy="unambiguous",
        primal_dual="primal",
        abs_ipm_opt_tol=1e-5,
    )
    dual_res, _ = state_exclusion(
        states,
        measurement="ppt",
        subsystems=[0],
        dimensions=[2],
        strategy="unambiguous",
        primal_dual="dual",
        abs_ipm_opt_tol=1e-5,
    )

    assert np.isclose(primal_res, 1 / np.sqrt(2), atol=5e-5)
    assert np.isclose(dual_res, 1 / np.sqrt(2), atol=5e-5)


def test_ppt_exclusion_requires_subsystems_and_dimensions():
    """Using `measurement='ppt'` without subsystems/dimensions should raise a clear ValueError."""
    with pytest.raises(ValueError, match="subsystems.*dimensions.*required"):
        state_exclusion(vectors=[bell(0), bell(1)], measurement="ppt")
