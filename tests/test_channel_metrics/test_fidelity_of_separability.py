"""Tests for channel Fidelity of Seperability."""

import numpy as np
import pytest

from toqito.states import max_mixed
from toqito.channel_metrics import fidelity_of_separability
from toqito.matrix_ops import tensor

purification_state = tensor(
    # System B:
    np.array([[1, 0], [0, 0]]),
    # System A:
    np.array([[1, 0], [0, 0]]),
    # System R:
    np.array([[0, 0], [0, 1]]),
)

bad_rho = 0.75 * tensor(
    # System B:
    np.array([[1, 0], [0, 0]]),
    # System A:
    np.array([[1, 0], [0, 0]]),
    # System R:
    np.array([[0, 0], [0, 1]]),
)

mixed_rho = max_mixed(8, is_sparse=False)


def test_errors_channel_SDP():
    """Tests for raised errors in channel SDP function."""
    with pytest.raises(ValueError, match="Provided input state is not a density matrix."):
        fidelity_of_separability(bad_rho, [2, 2, 2])
    with pytest.raises(AssertionError, match="For Channel SDP: require tripartite state dims."):
        fidelity_of_separability(purification_state, [2, 2])
    with pytest.raises(ValueError, match="This function only works for pure states."):
        fidelity_of_separability(mixed_rho, [2, 2, 2])


def test_sdp_output():
    """Test expected output of the SDP function."""
    channel_output_value = fidelity_of_separability(purification_state, [2, 2, 2], 2)
    assert np.isclose(1, channel_output_value)
