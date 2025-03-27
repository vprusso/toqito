"""Unit tests for the common_epistemic_overlap function."""

import numpy as np
import pytest

from toqito.state_props import common_epistemic_overlap


@pytest.mark.parametrize(
    "states, expected_overlap",
    [
        # Qubit test cases
        (
            # Orthogonal pure states (n=2)
            [np.array([1, 0]), np.array([0, 1])],  # |0>  # |1>
            0.0,
        ),
        (
            # Identical pure states (n=3)
            [
                np.array([[1, 0], [0, 0]]),  # |0><0| as density matrix
                np.array([1, 0]),  # |0> as state vector
                np.outer([1, 0], [1, 0]),  # |0><0| explicit construction
            ],
            1.0,
        ),
        (
            # Non-epistemic qubit triple (n=3)
            [
                np.array([[1, 0], [0, 0]]),  # |0><0|
                0.5 * np.array([[1, 1], [1, 1]]),  # |+><+|
                0.5 * np.array([[1, -1], [-1, 1]]),  # |-><-|
            ],
            0.0,
        ),
        ([np.eye(3) / 3] * 4, 1.0),  # Four identical maximally mixed states
        (
            [
                np.eye(3) / 3,  # Maximally mixed
                np.outer([1, 0, 0], [1, 0, 0]),  # Pure state |1>
            ],
            0.333333,  # 1/3 overlap from mixed state contribution
        ),
        # Composite dimension test cases
        ([np.eye(4) / 4] * 3, 1.0),  # Three identical maximally mixed states
        ([np.eye(30) / 6] * 2, 1.0),  # Two identical maximally mixed states
    ],
)
def test_epistemic_overlap_parametrized(states, expected_overlap):
    """Parametrized tests for core paper examples."""
    computed = common_epistemic_overlap(states)
    assert np.isclose(computed, expected_overlap, atol=1e-3)


@pytest.mark.parametrize("invalid_input, expected_error", [

    #non square density  matrix
    ([np.array([[1, 0], [0, 0], [0, 0]]), np.array([[1, 0], [0, 0], [0, 0]])], ValueError),

    #non consistent dimensions
    ([np.eye(2), np.eye(3)], ValueError),

])
def test_input_validation(invalid_input, expected_error):
    """Verify proper error handling for invalid inputs."""
    with pytest.raises(expected_error):
        common_epistemic_overlap(invalid_input)
