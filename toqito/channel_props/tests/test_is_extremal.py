"""Tests for the `is_extremal` function in the Toqito library."""

import numpy as np
import pytest

from toqito.channel_ops.kraus_to_choi import kraus_to_choi
from toqito.channel_props.is_extremal import is_extremal


# Test cases for unitary channels (which are extremal)
@pytest.mark.parametrize(
    "channel, expected",
    [
        # Direct list of Kraus operators
        ([np.array([[0, 1], [1, 0]])], True),
        # Dictionary representation with "kraus" key
        ({"kraus": [np.array([[0, 1], [1, 0]])]}, True),
    ],
)
def test_extremal_unitary_channel(channel, expected):
    """Verify that unitary channels are correctly identified as extremal."""
    assert is_extremal(channel) == expected


# Test cases for non-extremal channels
@pytest.mark.parametrize(
    "channel, expected",
    [
        (
            [
                np.sqrt(0.5) * np.array([[1, 0], [0, 1]]),
                np.sqrt(0.5) * np.array([[1, 0], [0, 1]]),
            ],
            False,
        ),
        (
            {
                "kraus": [
                    np.sqrt(0.5) * np.array([[1, 0], [0, 1]]),
                    np.sqrt(0.5) * np.array([[1, 0], [0, 1]]),
                ]
            },
            False,
        ),
    ],
)
def test_non_extremal_channel(channel, expected):
    """Verify that non-extremal channels are correctly identified."""
    assert is_extremal(channel) == expected


# Test cases where the channel is provided as a Choi matrix
@pytest.mark.parametrize(
    "channel, expected",
    [
        (kraus_to_choi([np.array([[0, 1], [1, 0]])]), True),
        ({"choi": kraus_to_choi([np.array([[0, 1], [1, 0]])])}, True),
    ],
)
def test_choi_input(channel, expected):
    """Verify that a channel provided as a Choi matrix is correctly processed."""
    assert is_extremal(channel) == expected


# Test example from Watrous's book (example 2.33)
@pytest.mark.parametrize(
    "channel, expected",
    [
        (
            [
                (1 / np.sqrt(6)) * np.array([[2, 0], [0, 1], [0, 1], [0, 0]]),
                (1 / np.sqrt(6)) * np.array([[0, 0], [1, 0], [1, 0], [0, 2]]),
            ],
            True,
        ),
    ],
)
def test_example_from_watrous(channel, expected):
    """Test the example 2.33 from Watrous's *Theory of Quantum Information*."""
    assert is_extremal(channel) == expected


# Test depolarizing channel, which is non-extremal for d > 2 (here d=2)
@pytest.mark.parametrize(
    "channel, expected",
    [
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
def test_depolarizing_channel(channel, expected):
    """Test that the depolarizing channel is correctly identified as non-extremal."""
    assert is_extremal(channel) == expected


def test_empty_kraus_operators():
    """Ensure an error is raised when the input is an empty list."""
    with pytest.raises(ValueError, match="The channel must contain at least one Kraus operator."):
        is_extremal([])


def test_invalid_dictionary_key():
    """Ensure an error is raised when the dictionary does not contain 'kraus' or 'choi'."""
    with pytest.raises(
        ValueError,
        match="Dictionary must have a 'kraus' or 'choi' key.",
    ):
        is_extremal({"invalid_key": np.array([[1, 0], [0, 1]])})


def test_unsupported_input_type():
    """Ensure an error is raised when the input type is not supported."""
    with pytest.raises(
        ValueError,
        match=(
            "Unsupported channel format. Provide Kraus operators, "
            "a Choi matrix, or a dictionary."
        ),
    ):
        is_extremal(42)  # Passing an integer instead of a valid quantum channel representation


def test_invalid_callable_object():
    """Ensure an error is raised when an object has a non-callable 'kraus' method."""
    class InvalidChannel:
        kraus = [np.array([[1, 0], [0, 1]])]  # kraus is a list, not a method

    with pytest.raises(
        ValueError,
        match=(
            "Unsupported channel format. Provide Kraus operators, "
            "a Choi matrix, or a dictionary."
        ),
    ):
        is_extremal(InvalidChannel())
