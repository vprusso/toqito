"""Test channel_distinguishability."""

import pytest

from toqito.channel_metrics import channel_distinguishability
from toqito.channel_ops import kraus_to_choi
from toqito.channels import amplitude_damping, dephasing, depolarizing, phase_damping

# Creating two amplitude damping channels.
amp_damp_1 = kraus_to_choi(amplitude_damping(gamma=0.22))
amp_damp_2 = kraus_to_choi(amplitude_damping(gamma=0.35))

# In Kraus representation.
amp_damp_1_kraus = amplitude_damping(gamma=0.22)
amp_damp_2_kraus = amplitude_damping(gamma=0.35)

# Creating two phase damping channels.
ph_damp_1 = kraus_to_choi(phase_damping(gamma=0.22))
ph_damp_2 = kraus_to_choi(phase_damping(gamma=0.35))


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
        # Same as previous channels but max(p) > 1.
        (amp_damp_1_kraus, amp_damp_2, [0.2, 1.8], [2, 2], 1),
    ],
)
def test_channel_distinguishability_bayesian(test_input_1, test_input_2, prior_prob, dim, expected):
    """Test function for Bayesian channel discrimination."""
    calculated_value = channel_distinguishability(test_input_1, test_input_2, prior_prob, dim)
    assert pytest.approx(expected, 1e-3) == calculated_value


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
    calculated_value = channel_distinguishability(
        test_input_1,
        test_input_2,
        prior_prob,
        dim,
        strategy="minimax",
        primal_dual=primal_dual,
    )
    assert pytest.approx(expected, 1e-3) == calculated_value


@pytest.mark.parametrize(
    "test_input_1, test_input_2, prior_prob, dim",
    [
        # Inconsistent dimensions between two channels.
        (
            depolarizing(4),
            dephasing(2),
            [0.5, 0.5],
            [2, 2],
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
def test_state_distinguishability_invalid_channels(test_input_1, test_input_2, prior_prob, dim, strategy):
    """Test function raises error for invalid channel dimensions for both bayesian and minimax settings."""
    with pytest.raises(
        ValueError,
        match="The channels must have the same dimension input and output spaces as each other.",
    ):
        channel_distinguishability(test_input_1, test_input_2, prior_prob, dim, strategy=strategy)


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


@pytest.mark.parametrize(
    "test_input1, test_input_2, prior_prob, dim, expected_msg",
    [
        # Sum of prior probabilities greater than 1.
        (
            dephasing(2),
            dephasing(2),
            [0.5, 0.9],
            [2, 2],
            "Sum of prior probabilities must add up to 1.",
        ),
        # Length of prior probability list not equal to two.
        (
            dephasing(2),
            dephasing(2),
            [0.5],
            [2, 2],
            "p must be a probability distribution with 2 entries.",
        ),
        # Prior probability must be provided for Bayesian strategy.
        (
            dephasing(2),
            dephasing(2),
            None,
            [2, 2],
            "Must provide valid prior probabilities for Bayesian strategy.",
        ),
    ],
)
def test_bayesian_channel_distinguishability_invalid_inputs(test_input1, test_input_2, prior_prob, dim, expected_msg):
    """Test function raises error as expected for invalid inputs for bayesian setting."""
    with pytest.raises(ValueError, match=expected_msg):
        channel_distinguishability(test_input1, test_input_2, prior_prob, dim)
