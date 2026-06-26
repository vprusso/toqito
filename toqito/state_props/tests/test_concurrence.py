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
        # Non-maximally entangled pure state cos(t)|00> + sin(t)|11> has concurrence |sin(2t)|.
        # For t = pi/6 this is sin(pi/3) = sqrt(3)/2.
        (
            (
                (np.cos(np.pi / 6) * np.kron(e_0, e_0) + np.sin(np.pi / 6) * np.kron(e_1, e_1))
                @ (np.cos(np.pi / 6) * np.kron(e_0, e_0) + np.sin(np.pi / 6) * np.kron(e_1, e_1)).conj().T
            ),
            np.sqrt(3) / 2,
        ),
        # The maximally mixed two-qubit state is separable, so its concurrence is zero.
        (np.identity(4) / 4, 0),
    ],
)
def test_concurrence(rho, expected_result):
    """Test function works as expected for a valid input."""
    res = concurrence(rho)
    np.testing.assert_allclose(res, expected_result, atol=1e-8)


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
