"""Test is_channel_antidistinguishable."""

import numpy as np
import pytest

from toqito.channel_props import is_channel_antidistinguishable
from toqito.channels import depolarizing
from toqito.state_props import is_antidistinguishable

PAULI_X = np.array([[0, 1], [1, 0]])
PAULI_Y = np.array([[0, -1j], [1j, 0]])
PAULI_Z = np.array([[1, 0], [0, -1]])


@pytest.mark.parametrize(
    "channels",
    [
        # The Choi states of the four Pauli channels are the four (antidistinguishable) Bell states.
        ([[np.eye(2)], [PAULI_X], [PAULI_Y], [PAULI_Z]]),
        # Three orthogonal Pauli channels remain antidistinguishable.
        ([[np.eye(2)], [PAULI_X], [PAULI_Z]]),
    ],
)
def test_is_channel_antidistinguishable_true(channels):
    """Antidistinguishable sets of channels are detected."""
    assert is_channel_antidistinguishable(channels)


@pytest.mark.parametrize(
    "channels",
    [
        # Two identical channels can never be antidistinguishable.
        ([depolarizing(2, 0.3), depolarizing(2, 0.3)]),
        # A pair of distinct but non-orthogonal channels is not antidistinguishable.
        ([depolarizing(2, 0.2), depolarizing(2, 0.4)]),
    ],
)
def test_is_channel_antidistinguishable_false(channels):
    """Non-antidistinguishable sets of channels are rejected."""
    assert not is_channel_antidistinguishable(channels)


def test_is_channel_antidistinguishable_atol_is_threaded():
    """The atol argument controls the zero test on the exclusion value."""
    # Two identical channels have exclusion value 0.5 (not antidistinguishable at any sane atol),
    # but a deliberately huge atol forces the value to be treated as zero.
    channels = [depolarizing(2, 0.3), depolarizing(2, 0.3)]
    assert not is_channel_antidistinguishable(channels)
    assert is_channel_antidistinguishable(channels, atol=0.6)


def test_is_channel_antidistinguishable_matches_choi_state_antidistinguishability():
    """For unitary channels, channel antidistinguishability matches Choi-state antidistinguishability."""
    unitaries = [np.eye(2), PAULI_X, PAULI_Y, PAULI_Z]
    choi_state_vecs = [U.reshape(-1) / np.sqrt(2) for U in unitaries]

    channel_result = is_channel_antidistinguishable([[U] for U in unitaries])
    state_result = bool(is_antidistinguishable(choi_state_vecs))
    assert channel_result == state_result
