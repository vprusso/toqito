"""Test concurrence."""

import numpy as np
import pytest

from toqito.state_props import concurrence
from toqito.states import bell

e_0, e_1 = np.array([[1], [0]]), np.array([[0], [1]])


@pytest.mark.parametrize(
    "rho, expected_result",
    [
        # Concurrence of maximally entangled Bell state.
        (bell(0) @ bell(0).conj().T, 1),
        # Concurrence of a product state is zero.
        (np.kron(e_0, e_1) @ np.kron(e_0, e_1).conj().T, 0),
    ],
)
def test_concurrence(rho, expected_result):
    """Test function works as expected for a valid input."""
    res = concurrence(rho)
    np.testing.assert_equal(np.isclose(res, expected_result), True)


@pytest.mark.parametrize(
    "rho",
    [
        # Tests for invalid dimension inputs.
        (np.identity(5)),
    ],
)
def test_concurrence_invalid_input(rho):
    """Test function works as expected for an invalid input."""
    with np.testing.assert_raises(ValueError):
        concurrence(rho)
