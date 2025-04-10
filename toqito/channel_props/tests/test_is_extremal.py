"""Tests for the `is_extremal` function in the Toqito library."""

import numpy as np
import pytest

from toqito.channel_ops.kraus_to_choi import kraus_to_choi
from toqito.channel_props.is_extremal import is_extremal


# Consolidated test for all extremal cases.
@pytest.mark.parametrize(
    "phi, expected",
    [
        # --- Unitary channels (extremal) ---
        # Flat list representation.
        ([np.array([[0, 1], [1, 0]])], True),
        # Nested list representation.
        ([[np.array([[0, 1], [1, 0]])]], True),
        # --- Non-extremal channels ---
        # Flat list representation: two identical Kraus operators.
        (
            [
                np.sqrt(0.5) * np.array([[1, 0], [0, 1]]),
                np.sqrt(0.5) * np.array([[1, 0], [0, 1]]),
            ],
            False,
        ),
        # Nested list representation of the above.
        (
            [
                [np.sqrt(0.5) * np.array([[1, 0], [0, 1]])],
                [np.sqrt(0.5) * np.array([[1, 0], [0, 1]])],
            ],
            False,
        ),
        # --- Choi matrix input ---
        (kraus_to_choi([np.array([[0, 1], [1, 0]])]), True),
        # --- Example from Watrous (Example 2.33) ---
        # Flat list representation.
        (
            [
                (1 / np.sqrt(6)) * np.array([[2, 0], [0, 1], [0, 1], [0, 0]]),
                (1 / np.sqrt(6)) * np.array([[0, 0], [1, 0], [1, 0], [0, 2]]),
            ],
            True,
        ),
        # Nested list representation.
        (
            [
                [(1 / np.sqrt(6)) * np.array([[2, 0], [0, 1], [0, 1], [0, 0]])],
                [(1 / np.sqrt(6)) * np.array([[0, 0], [1, 0], [1, 0], [0, 2]])],
            ],
            True,
        ),
        # --- Depolarizing channel (non-extremal) ---
        (
            [
                np.sqrt(1 - 3 * 0.75 / 4) * np.eye(2),
                np.sqrt(0.75 / 4) * np.array([[0, 1], [1, 0]]),
                np.sqrt(0.75 / 4) * np.array([[0, -1j], [1j, 0]]),
                np.sqrt(0.75 / 4) * np.array([[1, 0], [0, -1]]),
            ],
            False,
        ),
    ],
)
def test_is_extremal(phi, expected):
    """Test various cases for the `is_extremal` function."""
    assert is_extremal(phi) == expected


# <- Consolidated tests for ValueErrors. ->


@pytest.mark.parametrize(
    "phi, error_message",
    [
        # Empty list.
        ([], "The channel must contain at least one Kraus operator."),
        # Empty nested list.
        ([[]], "The channel must contain at least one Kraus operator."),
        # Invalid list contents.
        ([1, np.array([[0, 1], [1, 0]])], "Channel must be a list \\(or nested list\\) of Kraus operators."),
        # Unsupported input type.
        (42, "Channel must be a list of Kraus operators or a Choi matrix."),
    ],
)
def test_is_extremal_value_errors(phi, error_message):
    """Ensure ValueErrors are raised correctly for invalid inputs."""
    with pytest.raises(ValueError, match=error_message):
        is_extremal(phi)
