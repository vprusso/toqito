"""Test for the mutual coherence function."""

import numpy as np
import pytest

from toqito.matrix_props.mutual_coherence import mutual_coherence


@pytest.mark.parametrize(
    "setofvecs, expected_coherence",
    [
        # Identity matrix (mutual coherence should be 0).
        ([np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])], 0),
        # Matrix with repeated columns (mutual coherence should be 1).
        ([np.array([1, 0]), np.array([1, 0])], 1),
        # Random example with known coherence in 2x2.
        ([np.array([1, 0]), np.array([1, 1])], 1 / np.sqrt(2)),
        # Random example with known coherence in 3x3.
        ([np.array([1, 0, 1]), np.array([0, 1, 1]), np.array([1, 1, 0])], 1 / 2),
    ],
)
def test_mutual_coherence(setofvecs, expected_coherence):
    """Test the mutual coherence function for valid inputs."""
    result = mutual_coherence(setofvecs)
    assert np.isclose(result, expected_coherence, atol=1e-8)


@pytest.mark.parametrize(
    "setofvecs, expected_coherence",
    [
        # Input is not a list.
        (np.array([1, 0, 0]), TypeError),
        # Input is a list of non-1D arrays.
        ([np.array([[1, 0], [0, 1]])], ValueError),
    ],
)
def test_invalid_inputs(setofvecs, expected_coherence):
    """Test the mutual coherence function for invalid inputs."""
    with pytest.raises(expected_coherence):
        mutual_coherence(setofvecs)
