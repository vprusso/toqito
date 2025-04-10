"""Test random_states."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from toqito.rand import random_states
from toqito.state_props import is_pure


@pytest.mark.parametrize(
    "num_states, dim",
    [
        # Test with a single quantum state of dimension 2.
        (1, 2),
        # Test with multiple quantum states of the same dimension.
        (3, 2),
        # Test with a single quantum state of higher dimension.
        (1, 4),
        # Test with multiple quantum states of higher dimension.
        (2, 4),
    ],
)
def test_random_states(num_states, dim):
    """Test for random_states function."""
    # Generate a list of random quantum states.
    states = random_states(num_states, dim)

    # Ensure the number of states generated is as expected.
    assert len(states) == num_states

    # Check each state is a valid quantum state.
    for state in states:
        assert state.shape == (dim, 1)
        # Convert state vector to density matrix.
        dm = np.outer(state, np.conj(state))
        # Verify each state is pure.
        assert is_pure(dm)


@pytest.mark.parametrize(
    "num_states, dim, expected",
    [
        # Test with a single quantum state of dimension 2.
        (1, 2, [np.array([[-0.59005974 + 0.76831103j], [-0.2194029 + 0.11571532j]])]),
        # Test with multiple quantum states of the same dimension.
        (
            3,
            2,
            [
                np.array([[-0.73471584 - 0.47276295j], [-0.27319062 + 0.40256019j]]),
                np.array([[0.93422522 - 0.22964955j], [0.14070366 - 0.23385211j]]),
                np.array([[0.49063955 + 0.0518067j], [0.30769445 - 0.81358038j]]),
            ],
        ),
        # Test with a single quantum state of higher dimension.
        (
            1,
            4,
            [
                np.array(
                    [
                        [-0.45679821 + 0.42498307j],
                        [-0.16985205 + 0.26651935j],
                        [0.5947925 - 0.29393305j],
                        [0.0895817 + 0.25028557j],
                    ]
                )
            ],
        ),
        # Test with multiple quantum states of higher dimension.
        (
            2,
            4,
            [
                np.array(
                    [
                        [-0.42755142 - 0.13684957j],
                        [-0.15897716 - 0.13935391j],
                        [0.55671053 + 0.04200094j],
                        [0.08384617 - 0.65958915j],
                    ]
                ),
                np.array(
                    [
                        [0.4213706 + 0.5458888j],
                        [0.26425386 - 0.30728968j],
                        [-0.29143454 + 0.45801996j],
                        [0.24815808 + 0.06242098j],
                    ]
                ),
            ],
        ),
    ],
)
def test_seed(num_states, dim, expected):
    """Test that the function returns the expected output when seeded."""
    states = random_states(num_states, dim, seed=123)
    assert len(states) == len(expected)

    for state, expected_state in zip(states, expected):
        assert_allclose(state, expected_state)
