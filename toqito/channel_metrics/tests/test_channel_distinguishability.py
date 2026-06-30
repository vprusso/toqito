"""Test channel_distinguishability."""

import pytest

from toqito.channel_metrics import channel_distinguishability
from toqito.channel_ops import kraus_to_choi
from toqito.channels import amplitude_damping, dephasing, depolarizing, phase_damping

# Creating two amplitude damping channels.
amp_damp_1 = kraus_to_choi(amplitude_damping(gamma=0.22, return_kraus_ops=True))
amp_damp_2 = kraus_to_choi(amplitude_damping(gamma=0.35, return_kraus_ops=True))

# In Kraus representation.
amp_damp_1_kraus = amplitude_damping(gamma=0.22, return_kraus_ops=True)
amp_damp_2_kraus = amplitude_damping(gamma=0.35, return_kraus_ops=True)

# Creating two phase damping channels.
ph_damp_1 = kraus_to_choi(phase_damping(gamma=0.22, return_kraus_ops=True))
ph_damp_2 = kraus_to_choi(phase_damping(gamma=0.35, return_kraus_ops=True))


@pytest.mark.parametrize(
    "test_input_1, test_input_2, prior_prob, dim, expected",
    [
        # Distinguishing two identical channels.
        (dephasing(2), dephasing(2), [0.5, 0.5], [2, 2], 0.5),
        # Distinguishing two amplitude damping channels.
        (amp_damp_1, amp_damp_2, [0.2, 0.8], [2, 2], 0.8),
        # One channel in Kraus and another in Choi representation.
        (amp_damp_1_kraus, amp_damp_2, [0.2, 0.8], [2, 2], 0.8),
        # Both channels in Kraus representation.
        (amp_damp_1_kraus, amp_damp_2_kraus, [0.2, 0.8], [2, 2], 0.8),
        # Degenerate prior (certain which channel): trivially distinguishable.
        (amp_damp_1_kraus, amp_damp_2, [1.0, 0.0], [2, 2], 1),
        # Unnormalized weights are accepted and normalized internally ([2, 8] -> [0.2, 0.8]).
        (amp_damp_1, amp_damp_2, [2, 8], [2, 2], 0.8),
    ],
)
def test_channel_distinguishability_bayesian(test_input_1, test_input_2, prior_prob, dim, expected):
    """Test function for Bayesian channel discrimination."""
    calculated_value, operators = channel_distinguishability(test_input_1, test_input_2, prior_prob, dim)
    assert pytest.approx(expected, 1e-3) == calculated_value
    assert operators == []


@pytest.mark.parametrize(
    "test_input_1, test_input_2, prior_prob, dim, primal_dual, expected",
    [
        # Distinguishing two amplitude damping channels.
        (amp_damp_1, amp_damp_2, None, [2, 2], "dual", 0.55),
        # Distinguishing two phase damping channels.
        (ph_damp_1, ph_damp_2, None, [2, 2], "primal", 0.51),
    ],
)
def test_channel_distinguishability_minimax(test_input_1, test_input_2, prior_prob, dim, primal_dual, expected):
    """Test function for minimax channel discrimination."""
    calculated_value, operators = channel_distinguishability(
        test_input_1,
        test_input_2,
        prior_prob,
        dim,
        strategy="minimax",
        primal_dual=primal_dual,
    )
    assert pytest.approx(expected, 1e-3) == calculated_value
    # The minimax SDP branches return optimal strategy operators.
    assert len(operators) >= 1


@pytest.mark.parametrize(
    "test_input_1, test_input_2, prior_prob",
    [
        # Inconsistent dimensions between two channels.
        (
            depolarizing(4),
            dephasing(2),
            [0.5, 0.5],
        ),
    ],
)
@pytest.mark.parametrize(
    "strategy",
    [
        "Bayesian",
        "Minimax",
    ],
)
def test_state_distinguishability_invalid_channels(test_input_1, test_input_2, prior_prob, strategy):
    """Test function raises error for invalid channel dimensions for both bayesian and minimax settings."""
    with pytest.raises(
        ValueError,
        match="The channels must have the same dimension input and output spaces as each other.",
    ):
        channel_distinguishability(test_input_1, test_input_2, prior_prob, strategy=strategy)


@pytest.mark.parametrize(
    "test_input_1, test_input_2, prior_prob, dim",
    [
        (
            dephasing(2),
            dephasing(2),
            [0.5, 0.5],
            [2, 2],
        ),
    ],
)
@pytest.mark.parametrize(
    "strategy",
    [
        "Random",
    ],
)
def test_state_distinguishability_invalid_strategy(test_input_1, test_input_2, prior_prob, dim, strategy):
    """Test function raises error for strategy other than `Bayesian` or `Minimax`."""
    with pytest.raises(
        ValueError,
        match="The strategy must either be Bayesian or Minimax.",
    ):
        channel_distinguishability(test_input_1, test_input_2, prior_prob, dim, strategy=strategy)


def test_channel_distinguishability_invalid_primal_dual():
    """Invalid minimax formulation names should raise a clear ValueError."""
    with pytest.raises(ValueError, match="primal_dual option"):
        channel_distinguishability(
            dephasing(2),
            dephasing(2),
            None,
            [2, 2],
            strategy="minimax",
            primal_dual="bogus",
        )


@pytest.mark.parametrize(
    "test_input1, test_input_2, prior_prob, dim, expected_msg",
    [
        # Length of prior probability list not equal to two.
        (
            dephasing(2),
            dephasing(2),
            [0.5],
            [2, 2],
            "probs must be a probability distribution with 2 entries.",
        ),
        # Negative weights are rejected.
        (
            dephasing(2),
            dephasing(2),
            [-0.5, 1.5],
            [2, 2],
            "Prior probabilities must be non-negative.",
        ),
        # Weights summing to zero are rejected.
        (
            dephasing(2),
            dephasing(2),
            [0.0, 0.0],
            [2, 2],
            "Prior probabilities must have a positive sum.",
        ),
    ],
)
def test_bayesian_channel_distinguishability_invalid_inputs(test_input1, test_input_2, prior_prob, dim, expected_msg):
    """Test function raises error as expected for invalid inputs for bayesian setting."""
    with pytest.raises(ValueError, match=expected_msg):
        channel_distinguishability(test_input1, test_input_2, prior_prob, dim)


def test_bayesian_channel_distinguishability_uniform_default():
    """When probs is omitted, the Bayesian strategy uses a uniform prior."""
    value_default, _ = channel_distinguishability(dephasing(2), dephasing(2), None, [2, 2])
    value_uniform, _ = channel_distinguishability(dephasing(2), dephasing(2), [0.5, 0.5], [2, 2])
    assert pytest.approx(value_uniform, 1e-6) == value_default
