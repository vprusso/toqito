"""Test has_symmetric_extension."""

import numpy as np
import pytest

from toqito.state_props import has_symmetric_extension
from toqito.states import bell


@pytest.mark.parametrize(
    "rho, level, dim, ppt, expected_result",
    [
        # Check whether 2-qubit state has a symmetric extension.
        (
            np.array([[1, 0, 0, -1], [0, 1, 1 / 2, 0], [0, 1 / 2, 1, 0], [-1, 0, 0, 1]]),
            2,
            None,
            True,
            True,
        ),
        # Check whether state has level-1 symmetric extension."""
        (np.identity(4), 1, None, False, True),
        # Entangled state should not have symmetric extension for some level."""
        (bell(0) @ bell(0).conj().T, 1, None, True, False),
        # Entangled state should not have symmetric extension for some level (level-2).
        (bell(0) @ bell(0).conj().T, 2, None, True, False),
        # Provide dimension of system as list.
        (bell(0) @ bell(0).conj().T, 2, [2, 2], True, False),
        # Entangled state should not have non-PPT-symmetric extension for some level (level-2).
        (bell(0) @ bell(0).conj().T, 2, 2, False, False),
        # Entangled state should not have PPT-symmetric extension for some level (level-2)."""
        (bell(0) @ bell(0).conj().T, 2, 2, True, False),
    ],
)
def test_has_symmetric_extension(rho, level, dim, ppt, expected_result):
    """Test function works as expected for a valid input."""
    np.testing.assert_equal(has_symmetric_extension(rho=rho, level=level, dim=dim, ppt=ppt), expected_result)


@pytest.mark.parametrize(
    "rho, level, dim, ppt",
    [
        # Check whether 2-qubit state has a symmetric extension.
        (np.identity(6), 1, 4, True),
    ],
)
def test_has_symmetric_extension_invalid_dim(rho, level, dim, ppt):
    """Tests for invalid dimension inputs."""
    with np.testing.assert_raises(ValueError):
        rho = np.identity(6)
        has_symmetric_extension(rho=rho, level=level, dim=dim, ppt=ppt)
