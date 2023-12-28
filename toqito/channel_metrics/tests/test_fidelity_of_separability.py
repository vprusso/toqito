"""Tests for channel Fidelity of Seperability."""

import numpy as np
import pytest

from toqito.channel_metrics import fidelity_of_separability
from toqito.matrix_ops import tensor
from toqito.states import max_mixed

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


@pytest.mark.parametrize(
    "test_input, input_dim, expected_msg, expected_error",
    [
        (bad_rho, [2, 2, 2], "Provided input state is not a density matrix.", ValueError),
        (purification_state, [2, 2], "For Channel SDP: require tripartite state dims.", AssertionError),
        (mixed_rho, [2, 2, 2], "This function only works for pure states.", ValueError),
    ],
)
def test_errors_channel_SDP(test_input, input_dim, expected_msg, expected_error):
    """Tests for raised errors in channel SDP function."""
    with pytest.raises(expected_error, match=expected_msg):
        fidelity_of_separability(test_input, input_dim)


def test_sdp_output():
    """Test expected output of the SDP function."""
    channel_output_value = fidelity_of_separability(purification_state, [2, 2, 2], 2)
    assert np.isclose(1, channel_output_value)
