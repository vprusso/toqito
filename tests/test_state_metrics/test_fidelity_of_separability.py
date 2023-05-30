"""Tests for Fidelity of Seperability."""

import numpy as np
import pytest
from toqito.states import basis
from toqito.states import bell
from toqito.state_metrics import (
    state_ppt_extendible_fidelity, channel_ppt_extendible_fidelity)

# test mixed rho
e_0, e_1 = basis(2, 0), basis(2, 1)
mixed_density_matrix = 3 / 4 * e_0 * e_0.conj().T + 1 / 4 * e_1 * e_1.conj().T

# test pure rho
u = bell(0)
pure_density_matrix = u * u.conj().T

# test bad density matrix
bad_density_matrix = 1 / 2 * np.array([[1, 2], [3, 1]])


@pytest.mark.parametrize(
    "test_func", [
        state_ppt_extendible_fidelity, channel_ppt_extendible_fidelity]
)
@pytest.mark.parametrize("good_input", [
    mixed_density_matrix, pure_density_matrix])
def test_errors_raised(test_func, good_input):
    """Tests for riased errors."""
    with pytest.raises(
            AssertionError,
            match="Incorrect bipartite state dimensions provided."):
        test_func(good_input, [2, 2, 2])
    with pytest.raises(
            ValueError,
            match="Provided input state is not a density matrix."):
        test_func(bad_density_matrix, [2, 2])
