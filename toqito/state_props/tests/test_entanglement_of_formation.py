"""Test entanglement_of_formation."""

import numpy as np
import pytest

from toqito.state_props import entanglement_of_formation
from toqito.states import bell, max_mixed

e_0, e_1 = np.array([[1], [0]]), np.array([[0], [1]])


@pytest.mark.parametrize(
    "rho, dim, expected_result",
    [
        # The entanglement-of-formation on a Bell state.
        (bell(0) @ bell(0).conj().T, None, 1),
        # The entanglement-of-formation on a maximally mixed.
        (max_mixed(4, False) @ max_mixed(4, False).conj().T, None, 0),
        # The entanglement-of-formation on a Bell state with int dim
        (bell(0) @ bell(0).conj().T, 2, 1),
        # The entanglement-of-formation on a maximally mixed with int dim
        (max_mixed(4, False) @ max_mixed(4, False).conj().T, 1, 0),
        # The entanglement-of-formation on a Bell state with list dim
        (bell(0) @ bell(0).conj().T, [2, 2], 1),
        # The entanglement-of-formation on a maximally mixed with list dim
        (max_mixed(4, False) @ max_mixed(4, False).conj().T, [2, 2], 0),
    ],
)
def test_entanglement_of_formation(rho, dim, expected_result):
    """Test function works as expected for a valid input."""
    assert np.isclose(entanglement_of_formation(rho, dim), expected_result)


@pytest.mark.parametrize(
    "rho, dim, error_msg",
    [
        # Invalid local dimension for entanglement_of_formation.
        (np.identity(4), 3, "Invalid dimension: Please provide local dimensions that match the size of `rho`."),
        # Not presently known how to calculate for mixed states.
        (
            3 / 4 * e_0 @ e_0.conj().T + 1 / 4 * e_1 @ e_1.conj().T,
            None,
            "Invalid input: It is presently only known how to compute "
            "the entanglement-of-formation for two-qubit states and pure "
            "states.",
        ),
        # Invalid non-square matrix for entanglement_of_formation.
        (
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
            None,
            "Invalid dimension: `rho` must be either a vector or square matrix.",
        ),
        # The entanglement-of-formation on a maximally mixed with list dim
        (
            max_mixed(4, False) @ max_mixed(4, False).conj().T,
            [1, 1],
            "Invalid dimension: Please provide local dimensions that match the size of `rho`.",
        ),
    ],
)
def test_entanglement_of_formation_invalid(rho, dim, error_msg):
    """Ensures that an integer above 4 is error-checked."""
    with pytest.raises(ValueError, match=error_msg):
        entanglement_of_formation(rho, dim)
