"""Tests for Fidelity of Seperability."""

import numpy as np
import pytest
from toqito.states import bell, max_mixed
from toqito.state_metrics import (
    state_ppt_extendible_fidelity, channel_ppt_extendible_fidelity)
from toqito.matrix_ops import tensor


# test mixed rho
mixed_rho = 1/2 * bell(0) * bell(0).conj().T + 1 / 2 * bell(
    3) * bell(3).conj().T

# test pure entangled rho
u = bell(0)
entangled_rho = u * u.conj().T

# test bad density matrix
bad_rho = 1 / 2 * np.array([[1, 2], [3, 1]])

# test pure separable rho
sep_rho = np.kron(
        np.array([[1, 0], [0, 0]]),
        np.array([[0, 0], [0, 1]])
    )

purification_state = tensor(
        # System B:
        np.array([[1, 0], [0, 0]]),
        # System A:
        np.array([[1, 0], [0, 0]]),
        # System R:
        np.array([[0, 0], [0, 1]])
    )


def test_errors_state_SDP():
    """Tests for riased errors in state SDP function."""
    with pytest.raises(
            ValueError,
            match="Provided input state is not a density matrix."):
        state_ppt_extendible_fidelity(bad_rho, [2, 2])
    with pytest.raises(
            AssertionError,
            match="For State SDP: require bipartite state dims."):
        state_ppt_extendible_fidelity(sep_rho, [2, 2, 2])
    with pytest.raises(
            ValueError,
            match="Provided input state is entangled."):
        state_ppt_extendible_fidelity(entangled_rho, [2, 2])
    with pytest.raises(
            ValueError,
            match="This function only works for pure states."):
        state_ppt_extendible_fidelity(mixed_rho, [2, 2])


def test_sdp_output():
    """Test expected output of both the SDP functions."""
    expected_value = 1
    state_output_value = state_ppt_extendible_fidelity(sep_rho, [2, 2], 2)
    assert np.isclose(expected_value, state_output_value)

    channel_output_value = channel_ppt_extendible_fidelity(
        purification_state, [2, 2, 2], 2)
    assert np.isclose(expected_value, channel_output_value)


# re-define states for channel SDP function
bad_rho = 0.75*tensor(
        # System B:
        np.array([[1, 0], [0, 0]]),
        # System A:
        np.array([[1, 0], [0, 0]]),
        # System R:
        np.array([[0, 0], [0, 1]])
    )

mixed_rho = max_mixed(8, is_sparse=False)


def test_errors_channel_SDP():
    """Tests for riased errors in state SDP function."""
    with pytest.raises(
            ValueError,
            match="Provided input state is not a density matrix."):
        channel_ppt_extendible_fidelity(bad_rho, [2, 2, 2])
    with pytest.raises(
            AssertionError,
            match="For Channel SDP: require tripartite state dims."):
        channel_ppt_extendible_fidelity(purification_state, [2, 2])
    with pytest.raises(
            ValueError,
            match="This function only works for pure states."):
        channel_ppt_extendible_fidelity(mixed_rho, [2, 2, 2])
