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
            [
                np.array([1, 0]),                 # |0>
                np.array([0, 1])                  # |1>
            ],
            0.0
        ),
        (
            # Identical pure states (n=3)
            [
                np.array([[1, 0], [0, 0]]),       # |0><0| as density matrix
                np.array([1, 0]),                 # |0> as state vector
                np.outer([1, 0], [1, 0])          # |0><0| explicit construction
            ],
            1.0
        ),
        (
            # Non-epistemic qubit triple (n=3)
            [
                np.array([[1, 0], [0, 0]]),       # |0><0|
                0.5*np.array([[1, 1], [1, 1]]),   # |+><+|
                0.5*np.array([[1, -1], [-1, 1]])  # |-><-|
            ],
            0.0
        ),

        # Qutrit test cases (d=3)
        (
            # Fully non-epistemic case (n=4)
            [np.eye(3)/3]*4,                      # Four identical maximally mixed states
            0.0
        ),
        (
            # Mixed vs pure state
            [
                np.eye(3)/3,                      # Maximally mixed
                np.outer([0,1,0], [0,1,0])        # Pure state |1>
            ],
            0.333  # 1/3 overlap from mixed state contribution
        ),

        # Composite dimension test cases
        (
            # 4D system (d=4=2×2)
            [np.eye(4)/4]*3,                      # Three identical maximally mixed states
            1.0
        ),
        (
            [np.eye(6)/6]*2,                      # Two identical maximally mixed states
            1.0
        )
    ]
)
def test_epistemic_overlap_parametrized(states, expected_overlap):
    """Parametrized tests for core paper examples."""
    computed = common_epistemic_overlap(states)
    assert np.isclose(computed, expected_overlap, atol=1e-3)



def test_error_handling():
    """Test invalid input handling."""
    # Empty list
    with pytest.raises(ValueError):
        common_epistemic_overlap([])

    # Dimension mismatch
    with pytest.raises(ValueError):
        common_epistemic_overlap([np.eye(2), np.eye(3)])