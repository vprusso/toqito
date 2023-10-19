"""Tests for State Fidelity of Seperability."""

import numpy as np
import pytest

from toqito.matrix_ops import tensor
from toqito.state_metrics import fidelity_of_separability
from toqito.states import bell

# test mixed rho
mixed_rho = 1 / 2 * bell(0) @ bell(0).conj().T + 1 / 2 * bell(3) @ bell(3).conj().T

# test pure entangled rho
u = bell(0)
entangled_rho = u @ u.conj().T

# test bad density matrix
bad_rho = 1 / 2 * np.array([[1, 2], [3, 1]])

# test pure separable rho
sep_rho = np.kron(np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]]))

purification_state = tensor(
    # System B:
    np.array([[1, 0], [0, 0]]),
    # System A:
    np.array([[1, 0], [0, 0]]),
    # System R:
    np.array([[0, 0], [0, 1]]),
)


def test_errors_state_SDP():
    """Tests for raised errors in state SDP function."""
    with pytest.raises(ValueError, match="Provided input state is not a density matrix."):
        fidelity_of_separability(bad_rho, [2, 2])
    with pytest.raises(AssertionError, match="For State SDP: require bipartite state dims."):
        fidelity_of_separability(sep_rho, [2, 2, 2])
    with pytest.raises(ValueError, match="Provided input state is entangled."):
        fidelity_of_separability(entangled_rho, [2, 2])
    with pytest.raises(ValueError, match="This function only works for pure states."):
        fidelity_of_separability(mixed_rho, [2, 2])


def test_sdp_output():
    """Test expected output of the state SDP function."""
    state_output_value = fidelity_of_separability(sep_rho, [2, 2], 2)
    assert np.isclose(1, state_output_value)
