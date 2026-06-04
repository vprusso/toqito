"""Test channel_exclusion."""

import pytest

from toqito.channel_metrics import channel_exclusion
from toqito.channels import amplitude_damping, depolarizing


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


def test_channel_exclusion_invalid_probability_sum():
    """Probabilities must sum to 1 up to a small tolerance."""
    channels = [depolarizing(2, 0.2), depolarizing(2, 0.9)]

    with pytest.raises(ValueError, match="Prior probabilities must sum to 1 within tolerance."):
        channel_exclusion(channels=channels, probs=[0.7, 0.4])


def test_channel_exclusion_invalid_number_of_channels():
    """At least two channels are required."""
    with pytest.raises(ValueError, match="At least 2 channels are required for channel exclusion."):
        channel_exclusion(channels=[depolarizing(2, 0.2)], probs=[1.0])