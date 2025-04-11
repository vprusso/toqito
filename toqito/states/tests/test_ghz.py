"""Test ghz."""

import numpy as np
import pytest

from toqito.matrix_ops import tensor
from toqito.states import ghz

# Define standard basis vectors for qubits.
e_0, e_1 = np.array([[1], [0]]), np.array([[0], [1]])
# Pre-computed expected GHZ state for 3 qubits: 1/sqrt(2) (|000> + |111>)
ghz_2_3 = 1 / np.sqrt(2) * (tensor(e_0, e_0, e_0) + tensor(e_1, e_1, e_1))

# Define standard basis vectors for qudits in dimension 4.
e0_4 = np.array([[1], [0], [0], [0]])
e1_4 = np.array([[0], [1], [0], [0]])
e2_4 = np.array([[0], [0], [1], [0]])
e3_4 = np.array([[0], [0], [0], [1]])
# Pre-computed expected GHZ state for 7 qudits in C^4: 1/sqrt(30) (|0000000> + 2|1111111> + 3|2222222> + 4|3333333>)
ghz_4_7 = (
    1
    / np.sqrt(30)
    * (
        tensor(e0_4, e0_4, e0_4, e0_4, e0_4, e0_4, e0_4)
        + 2 * tensor(e1_4, e1_4, e1_4, e1_4, e1_4, e1_4, e1_4)
        + 3 * tensor(e2_4, e2_4, e2_4, e2_4, e2_4, e2_4, e2_4)
        + 4 * tensor(e3_4, e3_4, e3_4, e3_4, e3_4, e3_4, e3_4)
    )
)


@pytest.mark.parametrize(
    "dim, num_qubits, coeff, expected_state",
    [
        # Test the standard 3-qubit GHZ state using the default normalized coefficients.
        (2, 3, None, ghz_2_3),
        # Test the same 3-qubit GHZ state when non-normalized coefficients are provided.
        (2, 3, [2, 2], ghz_2_3),
        # Test the 7-qudit (dim=4) GHZ state with already normalized coefficients.
        (4, 7, np.array([1, 2, 3, 4]) / np.sqrt(30), ghz_4_7),
        # Test the same 7-qudit state when non-normalized coefficients are provided.
        (4, 7, [1, 2, 3, 4], ghz_4_7),
    ],
)
def test_ghz_valid_inputs(dim, num_qubits, coeff, expected_state):
    """Test that ghz returns the expected state regardless of input coefficient normalization."""
    result = ghz(dim, num_qubits, coeff)
    np.testing.assert_allclose(result, expected_state)
