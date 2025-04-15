"""Test for the mutual coherence function."""

import numpy as np
import pytest

from toqito.matrix_props.mutual_coherence import mutual_coherence


@pytest.mark.parametrize(
    "matrix, expected_coherence",
    [
        # Test case 1: Identity matrix (mutual coherence should be 0).
        (np.eye(3), 0),
        # Test case 2: Matrix with repeated columns (mutual coherence should be 1).
        (np.array([[1, 1], [0, 0]]), 1),
        # Test case 3:
        (np.array([[1, 0], [1, 1]]), 1/np.sqrt(2)),
        # Test case 4:
        (np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]]), 1 / 2),
    ],
)
def test_mutual_coherence(matrix, expected_coherence):
    """Test the mutual coherence function."""
    result = mutual_coherence(matrix)
    assert np.isclose(result, expected_coherence, atol=1e-8)
