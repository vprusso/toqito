"""Test channel_distinguishability."""

import numpy as np
import pytest

from toqito.channel_metrics import channel_distinguishability
from toqito.channel_ops import kraus_to_choi
from toqito.channels import amplitude_damping, dephasing, depolarizing, phase_damping
from toqito.matrices import pauli

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


@pytest.mark.parametrize("primal_dual", ["primal", "dual"])
def test_three_depolarizing_channels(primal_dual):
    """Discrimination among 3 depolarizing channels with distinct noise parameters."""
    channels = [depolarizing(2, 0.1), depolarizing(2, 0.5), depolarizing(2, 0.9)]
    value, operators = channel_distinguishability(channels, probs=[1 / 3, 1 / 3, 1 / 3], primal_dual=primal_dual)
    # A uniformly random guess succeeds with probability 1/3; the SDP must do at least as well.
    assert 1 / 3 <= value <= 1
    if primal_dual == "primal":
        assert len(operators) == 3
    else:
        assert len(operators) == 1


def test_three_depolarizing_channels_primal_equals_dual():
    """The primal and dual SDP values agree for 3 depolarizing channels."""
    channels = [depolarizing(2, 0.1), depolarizing(2, 0.5), depolarizing(2, 0.9)]
    value_primal, _ = channel_distinguishability(channels, primal_dual="primal")
    value_dual, _ = channel_distinguishability(channels, primal_dual="dual")
    assert pytest.approx(value_dual, abs=1e-6) == value_primal


@pytest.mark.parametrize("num_channels", [3, 4])
@pytest.mark.parametrize("primal_dual", ["primal", "dual"])
def test_orthogonal_unitary_channels_perfectly_distinguishable(num_channels, primal_dual):
    """Pauli unitary channels are orthogonal in Hilbert-Schmidt inner product, hence perfectly distinguishable."""
    channels = [[pauli(ind)] for ind in range(num_channels)]
    value, _ = channel_distinguishability(channels, primal_dual=primal_dual)
    assert pytest.approx(1, abs=1e-6) == value


@pytest.mark.parametrize("prior_prob", [None, [0.5, 0.5], [0.2, 0.8], [2, 8]])
@pytest.mark.parametrize("primal_dual", ["primal", "dual"])
def test_two_channel_list_mode_matches_cb_trace_norm(prior_prob, primal_dual):
    """The n = 2 list-mode SDP agrees with the legacy CB-trace-norm shortcut, including non-uniform priors."""
    value_legacy, _ = channel_distinguishability(amp_damp_1, amp_damp_2, prior_prob)
    value_sdp, operators = channel_distinguishability(
        [amp_damp_1, amp_damp_2], probs=prior_prob, primal_dual=primal_dual
    )
    assert pytest.approx(value_legacy, abs=1e-6) == value_sdp
    assert len(operators) >= 1


def test_list_mode_accepts_mixed_kraus_and_choi():
    """List mode accepts each channel either as a Choi matrix or as a Kraus operator list."""
    value_choi, _ = channel_distinguishability([amp_damp_1, amp_damp_2, dephasing(2)])
    value_mixed, _ = channel_distinguishability([amp_damp_1_kraus, amp_damp_2, dephasing(2)])
    assert pytest.approx(value_choi, abs=1e-6) == value_mixed


def test_list_mode_minimax_two_channels():
    """List mode with 2 channels routes minimax to the existing helpers."""
    value_legacy, _ = channel_distinguishability(amp_damp_1, amp_damp_2, None, [2, 2], strategy="minimax")
    value_list, operators = channel_distinguishability([amp_damp_1, amp_damp_2], strategy="minimax")
    assert pytest.approx(value_legacy, abs=1e-6) == value_list
    assert len(operators) >= 1

    value_legacy_primal, _ = channel_distinguishability(
        amp_damp_1, amp_damp_2, None, [2, 2], strategy="minimax", primal_dual="primal"
    )
    value_list_primal, _ = channel_distinguishability(
        [amp_damp_1, amp_damp_2], strategy="minimax", primal_dual="primal"
    )
    assert pytest.approx(value_legacy_primal, abs=1e-6) == value_list_primal


def test_list_mode_minimax_three_channels_raises():
    """Minimax discrimination is out of scope for more than 2 channels."""
    channels = [depolarizing(2, 0.1), depolarizing(2, 0.5), depolarizing(2, 0.9)]
    with pytest.raises(ValueError, match="Minimax discrimination is only supported for exactly 2 channels."):
        channel_distinguishability(channels, strategy="minimax")


def test_list_mode_degenerate_prior_short_circuit():
    """A degenerate prior in list mode short-circuits to a certain outcome."""
    channels = [depolarizing(2, 0.1), depolarizing(2, 0.5), depolarizing(2, 0.9)]
    value, operators = channel_distinguishability(channels, probs=[0, 1, 0])
    assert value == 1.0
    assert operators == []


@pytest.mark.parametrize(
    "channels, probs, expected_msg",
    [
        # A single channel is not enough for a discrimination task.
        ([depolarizing(2, 0.1)], None, "When psi is None, phi must be a list of at least 2 channels."),
        # A bare Choi matrix (not a list) with psi=None is rejected.
        (depolarizing(2, 0.1), None, "When psi is None, phi must be a list of at least 2 channels."),
        # Mismatched channel dimensions.
        (
            [depolarizing(2, 0.1), depolarizing(3, 0.5)],
            None,
            "The channels must have the same dimension input and output spaces as each other.",
        ),
        # Number of priors must match the number of channels.
        (
            [depolarizing(2, 0.1), depolarizing(2, 0.5), depolarizing(2, 0.9)],
            [0.5, 0.5],
            "probs must be a probability distribution with 3 entries.",
        ),
        # Negative weights are rejected.
        (
            [depolarizing(2, 0.1), depolarizing(2, 0.5), depolarizing(2, 0.9)],
            [-0.5, 1.0, 0.5],
            "Prior probabilities must be non-negative.",
        ),
        # Weights summing to zero are rejected.
        (
            [depolarizing(2, 0.1), depolarizing(2, 0.5), depolarizing(2, 0.9)],
            [0.0, 0.0, 0.0],
            "Prior probabilities must have a positive sum.",
        ),
    ],
)
def test_list_mode_invalid_inputs(channels, probs, expected_msg):
    """List mode raises clear errors for invalid channel collections and priors."""
    with pytest.raises(ValueError, match=expected_msg):
        channel_distinguishability(channels, probs=probs)


def test_list_mode_dual_operator_shape():
    """The dual SDP returns a single dual operator of the expected shape."""
    channels = [depolarizing(2, 0.1), depolarizing(2, 0.5), depolarizing(2, 0.9)]
    _, operators = channel_distinguishability(channels, primal_dual="dual")
    assert len(operators) == 1
    assert operators[0].shape == (4, 4)
    assert isinstance(operators[0], np.ndarray)
