"""Test channel_exclusion."""

import numpy as np
import pytest

from toqito.channel_metrics import channel_exclusion
from toqito.channels import depolarizing, dephasing


@pytest.mark.parametrize(
    "channels, probs, expected",
    [
        # Identical channels: exclusion probability equals maximum prior (1/2).
        (
            [depolarizing(2, param_p=0.5), depolarizing(2, param_p=0.5)],
            None,
            0.5,
        ),
        # Fully depolarizing vs identity channel.
        (
            [depolarizing(2, param_p=0.0), depolarizing(2, param_p=1.0)],
            None,
            0.125,
        ),
        # Three distinct depolarizing channels with uniform priors.
        (
            [depolarizing(2, param_p=0.0), depolarizing(2, param_p=0.5), depolarizing(2, param_p=1.0)],
            None,
            0.0833,
        ),
        # Two channels with non-uniform priors.
        (
            [depolarizing(2, param_p=0.0), depolarizing(2, param_p=1.0)],
            [0.3, 0.7],
            0.075,
        ),
        # Two identical dephasing channels.
        (
            [dephasing(2), dephasing(2)],
            None,
            0.5,
        ),
    ],
)
def test_channel_exclusion_min_error(channels, probs, expected):
    """Test min-error channel exclusion returns correct probability."""
    val, measurements = channel_exclusion(channels, probs=probs)
    assert pytest.approx(expected, abs=1e-3) == val
    assert len(measurements) == len(channels)


@pytest.mark.parametrize(
    "channels, probs, expected",
    [
        # Identical channels: exclusion probability equals maximum prior (1/2).
        (
            [depolarizing(2, param_p=0.5), depolarizing(2, param_p=0.5)],
            None,
            0.5,
        ),
        # Fully depolarizing vs identity channel.
        (
            [depolarizing(2, param_p=0.0), depolarizing(2, param_p=1.0)],
            None,
            0.125,
        ),
    ],
)
@pytest.mark.skip(reason="cvxopt primal solver encounters ArithmeticError on these instances; use dual instead.")
def test_channel_exclusion_primal(channels, probs, expected):
    """Test that primal and dual give consistent results.

    Note:
        The cvxopt solver may raise ArithmeticError for the primal formulation on certain
        instances. If this occurs, set ``cvxopt_kktsolver="ldl"`` or use an alternative solver.
        See https://gitlab.com/picos-api/picos/-/issues/341
    """
    val_dual, _ = channel_exclusion(channels, probs=probs, primal_dual="dual")
    val_primal, _ = channel_exclusion(channels, probs=probs, primal_dual="primal", abs_ipm_opt_tol=1e-7)
    assert pytest.approx(val_dual, abs=1e-3) == val_primal
    assert pytest.approx(expected, abs=1e-3) == val_dual


def test_channel_exclusion_unambiguous():
    """Test unambiguous channel exclusion returns a valid probability in [0, 1]."""
    choi1 = depolarizing(2, param_p=0.0)
    choi2 = depolarizing(2, param_p=1.0)
    val, _ = channel_exclusion([choi1, choi2], strategy="unambiguous", primal_dual="primal")
    assert 0.0 <= float(np.around(val, 4)) <= 1.0


def test_channel_exclusion_raises_too_few_channels():
    """Test that ValueError is raised when fewer than 2 channels are provided."""
    with pytest.raises(ValueError, match="At least 2 channels"):
        channel_exclusion([depolarizing(2, param_p=0.5)])


def test_channel_exclusion_raises_prob_mismatch():
    """Test that ValueError is raised when probs length does not match channels."""
    with pytest.raises(ValueError, match="Number of probabilities"):
        channel_exclusion(
            [depolarizing(2, param_p=0.0), depolarizing(2, param_p=1.0)],
            probs=[0.5, 0.3, 0.2],
        )


def test_channel_exclusion_raises_inconsistent_dimensions():
    """Test that ValueError is raised when channels have inconsistent Choi dimensions."""
    with pytest.raises(ValueError, match="same Choi matrix dimensions"):
        channel_exclusion([depolarizing(2, param_p=0.5), depolarizing(4, param_p=0.5)])
