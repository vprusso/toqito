"""Test schmidt_rank."""

import numpy as np
import pytest

from toqito.state_props import schmidt_rank
from toqito.states import bell

e_0, e_1 = np.array([[1], [0]]), np.array([[0], [1]])


@pytest.mark.parametrize(
    "rho, dim, expected_result",
    [
        # Computing the Schmidt rank of the entangled Bell state should yield a value greater than 1.
        (bell(0), None, 2),
        # Computing the Schmidt rank of Bell state with dim 1.
        (bell(0), 1, 1),
        # Computing the Schmidt rank of Bell state with list as argument for dims.
        (bell(0), [2, 2], 2),
        # Computing the Schmidt rank of a separable state should yield a value equal to 1.
        (
            1 / 2 * (np.kron(e_0, e_0) - np.kron(e_0, e_1) - np.kron(e_1, e_0) + np.kron(e_1, e_1)),
            None,
            1,
        ),
        # Computing Schmidt rank of separable density matrix should be 1.
        (np.identity(4), None, 1),
        # Computing Schmidt rank of separable density matrix should be 1.
        (np.identity(16), None, 1),
        # Computing Schmidt rank of first Bell density matrices should be 4.
        (bell(0) @ bell(0).conj().T, None, 4),
        # Computing Schmidt rank of second Bell density matrices should be 4.
        (bell(1) @ bell(1).conj().T, None, 4),
        # Computing Schmidt rank of third Bell density matrices should be 4.
        (bell(2) @ bell(2).conj().T, None, 4),
        # Computing Schmidt rank of fourth Bell density matrices should be 4.
        (bell(3) @ bell(3).conj().T, None, 4),
        # Computing Schmidt rank of first Bell density matrices should be 4 (dimension as integer).
        (bell(0) @ bell(0).conj().T, 2, 4),
        # Computing Schmidt rank of first Bell density matrices should be 4 (dimension as list).
        (bell(0) @ bell(0).conj().T, [2, 2], 4),
    ],
)
def test_schmidt_rank_bell_state(rho, dim, expected_result):
    """Test function works as expected for a valid input."""
    np.testing.assert_allclose(schmidt_rank(rho, dim), expected_result)


def test_schmidt_rank_entangled_state():
    """Computing Schmidt rank of entangled state should be > 1."""
    phi = (
        (1 + np.sqrt(6)) / (2 * np.sqrt(6)) * np.kron(e_0, e_0)
        + (1 - np.sqrt(6)) / (2 * np.sqrt(6)) * np.kron(e_0, e_1)
        + (np.sqrt(2) - np.sqrt(3)) / (2 * np.sqrt(6)) * np.kron(e_1, e_0)
        + (np.sqrt(2) + np.sqrt(3)) / (2 * np.sqrt(6)) * np.kron(e_1, e_1)
    )
    np.testing.assert_equal(schmidt_rank(phi) == 2, True)


def test_schmidt_rank_singlet_state():
    """Computing the Schmidt rank of the entangled singlet state should yield a value greater than 1."""
    rho = 1 / np.sqrt(2) * (np.kron(e_0, e_1) - np.kron(e_1, e_0))
    np.testing.assert_equal(schmidt_rank(rho) > 1, True)
