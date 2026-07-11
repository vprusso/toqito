"""Test channel_exclusion."""

import numpy as np
import pytest

from toqito.channel_metrics import completely_bounded_trace_norm
from toqito.channel_ops import kraus_to_choi
from toqito.channel_opt import channel_exclusion
from toqito.channels import amplitude_damping, depolarizing
from toqito.state_opt import state_exclusion


@pytest.mark.parametrize("primal_dual", ["primal", "dual"])
def test_channel_exclusion_identical_channels(primal_dual):
    """Identical channels cannot be excluded better than random guessing."""
    channels = [depolarizing(2, 0.3), depolarizing(2, 0.3)]

    value, strategy_ops = channel_exclusion(
        channels=channels,
        probs=[0.5, 0.5],
        primal_dual=primal_dual,
        cvxopt_kktsolver="ldl",
    )

    assert abs(value - 0.5) <= 1e-6
    assert len(strategy_ops) == 2


def test_channel_exclusion_primal_dual_agree_mixed_representations():
    """Primal and dual values should agree when channels are passed in mixed representations."""
    channels = [amplitude_damping(gamma=0.15), depolarizing(2, 0.65)]

    primal_value, _ = channel_exclusion(
        channels=channels,
        probs=[0.500000001, 0.499999999],
        primal_dual="primal",
        cvxopt_kktsolver="ldl",
    )
    dual_value, _ = channel_exclusion(
        channels=channels,
        probs=[0.500000001, 0.499999999],
        primal_dual="dual",
        cvxopt_kktsolver="ldl",
    )

    assert 0 <= primal_value <= 1
    assert 0 <= dual_value <= 1
    assert abs(primal_value - dual_value) <= 1e-5


def test_channel_exclusion_accepts_unnormalized_weights():
    """Unnormalized weights are accepted and normalized internally (matching state_exclusion)."""
    channels = [depolarizing(2, 0.2), depolarizing(2, 0.9)]

    val_unnormalized, _ = channel_exclusion(channels=channels, probs=[2.0, 2.0])
    val_normalized, _ = channel_exclusion(channels=channels, probs=[0.5, 0.5])
    assert val_unnormalized == pytest.approx(val_normalized, abs=1e-6)


def test_channel_exclusion_invalid_probability_sum():
    """Weights summing to zero are rejected."""
    channels = [depolarizing(2, 0.2), depolarizing(2, 0.9)]

    with pytest.raises(ValueError, match="Prior probabilities must have a positive sum."):
        channel_exclusion(channels=channels, probs=[0.0, 0.0])


def test_channel_exclusion_invalid_number_of_channels():
    """At least two channels are required."""
    with pytest.raises(ValueError, match="At least 2 channels are required for channel exclusion."):
        channel_exclusion(channels=[depolarizing(2, 0.2)], probs=[1.0])


@pytest.mark.parametrize(
    "channels,probs,strategy,primal_dual,match",
    [
        (
            [depolarizing(2, 0.2), depolarizing(2, 0.3)],
            [0.5, 0.5],
            "invalid_strategy",
            "primal",
            "The strategy must be either 'min_error' or 'unambiguous'.",
        ),
        (
            [depolarizing(2, 0.2), depolarizing(2, 0.3)],
            [0.5, 0.5],
            "min_error",
            "invalid",
            "The primal_dual option must be either 'primal' or 'dual'.",
        ),
        (
            [depolarizing(2, 0.2), depolarizing(2, 0.3)],
            [0.5],
            "min_error",
            "primal",
            "The number of probabilities must match the number of channels.",
        ),
        (
            [depolarizing(2, 0.2), depolarizing(2, 0.3)],
            [1.1, -0.1],
            "min_error",
            "primal",
            "Prior probabilities must be non-negative.",
        ),
        (
            [depolarizing(2, 0.2), depolarizing(3, 0.3)],
            [0.5, 0.5],
            "min_error",
            "primal",
            "All channels must have the same input and output dimensions.",
        ),
        (
            [depolarizing(2, 0.2), depolarizing(2, 0.3)],
            [0.5, 0.5],
            "unambiguous",
            "dual",
            "Unambiguous strategy supports only the primal formulation for Phase 2.",
        ),
    ],
)
def test_channel_exclusion_validation_branches(channels, probs, strategy, primal_dual, match):
    """Validation branches should reject invalid options and malformed inputs."""
    with pytest.raises(ValueError, match=match):
        channel_exclusion(channels=channels, probs=probs, strategy=strategy, primal_dual=primal_dual)


def test_unambiguous_exclusion_orthogonal_unitaries():
    """Orthogonal unitary channels should be perfectly excludable (inconclusive prob = 0)."""
    # Pauli X and Z
    X = np.array([[0.0, 1.0], [1.0, 0.0]])
    Z = np.array([[1.0, 0.0], [0.0, -1.0]])

    # Provide as Kraus lists so kraus_to_choi is invoked
    value, ops = channel_exclusion(
        channels=[[X], [Z]],
        probs=[0.5, 0.5],
        strategy="unambiguous",
        primal_dual="primal",
        cvxopt_kktsolver="ldl",
    )
    assert abs(value - 0.0) <= 1e-6
    # Expect the returned ops to include W_inc as the last element
    assert len(ops) == 3


def test_three_depolarizing_interpolation():
    """Three identical depolarizing channels give error 1/3; more distinct channels give lower error."""
    channels_identical = [depolarizing(2, 0.3), depolarizing(2, 0.3), depolarizing(2, 0.3)]
    val_identical, _ = channel_exclusion(
        channels=channels_identical,
        probs=[1 / 3, 1 / 3, 1 / 3],
        primal_dual="primal",
        cvxopt_kktsolver="ldl",
    )
    assert abs(val_identical - 1 / 3) <= 1e-6

    channels_distinct = [depolarizing(2, 0.0), depolarizing(2, 0.5), depolarizing(2, 1.0)]
    val_distinct, _ = channel_exclusion(
        channels=channels_distinct,
        probs=[1 / 3, 1 / 3, 1 / 3],
        primal_dual="primal",
        cvxopt_kktsolver="ldl",
    )
    assert 0 <= val_distinct <= val_identical


def test_orthogonal_unitaries_min_error_matches_state_exclusion():
    """For orthogonal unitaries, channel exclusion should match state_exclusion on normalized Choi states."""
    eye = np.eye(2)
    Z = np.array([[1.0, 0.0], [0.0, -1.0]])

    choi_I = kraus_to_choi([eye])
    choi_Z = kraus_to_choi([Z])

    # Normalize Choi matrices to density matrices for state_exclusion
    dim = 2
    rho_I = choi_I / dim
    rho_Z = choi_Z / dim

    chan_val, _ = channel_exclusion(
        channels=[choi_I, choi_Z],
        probs=[0.5, 0.5],
        primal_dual="primal",
        cvxopt_kktsolver="ldl",
    )
    # Use the dual formulation with LDL KKT solver for numerical stability.
    state_val, _ = state_exclusion([rho_I, rho_Z], probs=[0.5, 0.5], primal_dual="dual", cvxopt_kktsolver="ldl")

    assert abs(chan_val - state_val) <= 1e-6


def test_two_channel_exclusion_relates_to_diamond_norm():
    """For n=2 the exclusion error relates to the diamond norm of p0*Phi0 - p1*Phi1."""
    # Use two simple depolarizing channels with unequal priors
    choi_0 = depolarizing(2, 0.2)
    choi_1 = depolarizing(2, 0.7)

    p0 = 0.6
    p1 = 0.4

    # Compute diamond norm of the weighted difference
    cb_norm = completely_bounded_trace_norm(p0 * choi_0 - p1 * choi_1, "cvxopt", cvxopt_kktsolver="ldl")

    # For our helper, the two-hypothesis exclusion error equals 1/2*(1 - cb_norm).
    predicted_exclusion = 0.5 * (1 - cb_norm)

    chan_val, _ = channel_exclusion(
        channels=[choi_0, choi_1], probs=[p0, p1], primal_dual="primal", cvxopt_kktsolver="ldl"
    )

    assert abs(chan_val - predicted_exclusion) <= 1e-5
