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
            0.5,
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


@pytest.mark.parametrize(
    "states, expected_overlap_uppeder_bound",
    [
        # test cases based on the formula: epistemic overlap <= quantum overlap = 1 - 0.5 * tr|ρ₁ - ρ₂|
        (
            # Two diagonal density matrices
            [
                np.diag([0.7, 0.3]),
                np.diag([0.4, 0.6]),
            ],
            0.7,  # trace norm of difference = 0.6 -> overlap = 1 - 0.5*0.6 = 0.7
        ),
        (
            # Two non-orthogonal pure states: |0> and |+> = 1/√2 [1, 1]
            [
                np.array([1, 0]),
                np.array([1, 1]) / np.sqrt(2),
            ],
            0.5,  # 0.5 * tr|ρ₁ - ρ₂| = 1 -> overlap = 0.5
        ),
    ],
)
def test_epistemic_overlap_inequality(states, expected_overlap_uppeder_bound):
    """Test that common_epistemic_overlap returns the expected upper bound."""
    computed = common_epistemic_overlap(states)
    assert round(computed, 4) <= expected_overlap_uppeder_bound


@pytest.mark.parametrize(
    "states, expected_msg",
    [
        # Non-square density matrix should raise ValueError.
        ([np.array([[1, 2, 3], [4, 5, 6]])], r"Input must be either a vector or a square density matrix."),
        # States with inconsistent dimensions should raise ValueError.
        ([np.array([1, 0]), np.array([1, 0, 0])], r"All states must have consistent dimension"),
    ],
)
def test_common_epistemic_overlap_invalid_inputs(states, expected_msg):
    """Test that common_epistemic_overlap raises errors for invalid inputs."""
    with pytest.raises(ValueError, match=expected_msg):
        common_epistemic_overlap(states)
