"""Tests for the `is_extremal` function in the Toqito library."""

import numpy as np
import pytest

from toqito.channel_ops.kraus_to_choi import kraus_to_choi
from toqito.channel_props.is_extremal import is_extremal


# Test cases for unitary channels (which are extremal)
@pytest.mark.parametrize(
    "phi, expected",
    [
        # Direct flat list of Kraus operators
        ([np.array([[0, 1], [1, 0]])], True),
        # Nested list of Kraus operators
        ([[np.array([[0, 1], [1, 0]])]], True),
    ],
)
def test_extremal_unitary_channel(phi, expected):
    """Verify that unitary channels are correctly identified as extremal."""
    assert is_extremal(phi) == expected


# Test cases for non-extremal channels
@pytest.mark.parametrize(
    "phi, expected",
    [
        (
            # Flat list: two identical Kraus operators.
            [
                np.sqrt(0.5) * np.array([[1, 0], [0, 1]]),
                np.sqrt(0.5) * np.array([[1, 0], [0, 1]]),
            ],
            False,
        ),
        (
            # Nested list version of the above.
            [
                [np.sqrt(0.5) * np.array([[1, 0], [0, 1]])],
                [np.sqrt(0.5) * np.array([[1, 0], [0, 1]])],
            ],
            False,
        ),
    ],
)
def test_non_extremal_channel(phi, expected):
    """Verify that non-extremal channels are correctly identified."""
    assert is_extremal(phi) == expected


# Test cases where the channel is provided as a Choi matrix
@pytest.mark.parametrize(
    "phi, expected",
    [
        (kraus_to_choi([np.array([[0, 1], [1, 0]])]), True),
    ],
)
def test_choi_input(phi, expected):
    """Verify that a channel provided as a Choi matrix is correctly processed."""
    assert is_extremal(phi) == expected


# Test example from Watrous's book (example 2.33)
@pytest.mark.parametrize(
    "phi, expected",
    [
        (
            # Flat list representation.
            [
                (1 / np.sqrt(6)) * np.array([[2, 0], [0, 1], [0, 1], [0, 0]]),
                (1 / np.sqrt(6)) * np.array([[0, 0], [1, 0], [1, 0], [0, 2]]),
            ],
            True,
        ),
        (
            # Nested list representation.
            [
                [(1 / np.sqrt(6)) * np.array([[2, 0], [0, 1], [0, 1], [0, 0]])],
                [(1 / np.sqrt(6)) * np.array([[0, 0], [1, 0], [1, 0], [0, 2]])],
            ],
            True,
        ),
    ],
)
def test_example_from_watrous(phi, expected):
    """Test the example 2.33 from Watrous's *Theory of Quantum Information*."""
    assert is_extremal(phi) == expected


# Test depolarizing channel, which is non-extremal for d > 2 (here d=2)
@pytest.mark.parametrize(
    "phi, expected",
    [
        (
            # Flat list representation of depolarizing channel Kraus ops.
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
def test_depolarizing_channel(phi, expected):
    """Test that the depolarizing channel is correctly identified as non-extremal."""
    assert is_extremal(phi) == expected


# <- Tests for ValueError conditions ->

def test_empty_kraus_operators():
    """Ensure an error is raised when the input is an empty list."""
    with pytest.raises(ValueError, match="The channel must contain at least one Kraus operator."):
        is_extremal([])

def test_empty_nested_list():
    """Ensure an error is raised when a nested list contains no Kraus operators."""
    with pytest.raises(ValueError, match="The channel must contain at least one Kraus operator."):
        is_extremal([[]])

def test_invalid_list_contents():
    """Ensure an error is raised when the input list contains invalid elements."""
    with pytest.raises(ValueError, match="Channel must be a list \\(or nested list\\) of Kraus operators."):
        is_extremal([1, np.array([[0, 1], [1, 0]])])

def test_unsupported_input_type():
    """Ensure an error is raised when the input type is not supported."""
    with pytest.raises(ValueError, match="Channel must be a list of Kraus operators or a Choi matrix."):
        is_extremal(42)  # Passing an integer
