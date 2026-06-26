"""Test for the mutual coherence function."""

import numpy as np
import pytest

from toqito.matrix_props.mutual_coherence import mutual_coherence


@pytest.mark.parametrize(
    "vectors, expected_coherence",
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
def test_mutual_coherence(vectors, expected_coherence):
    """Test the mutual coherence function for valid inputs."""
    result = mutual_coherence(vectors)
    assert np.isclose(result, expected_coherence, atol=1e-8)


@pytest.mark.parametrize(
    "vectors, exception, expected_msg",
    [
        (None, TypeError, r"Input must be a list of 1D numpy arrays\."),
        ([1, 2, 3], ValueError, r"All elements in the list must be 1D numpy arrays\."),
        (
            [np.array([[1, 2], [3, 4]]), np.array([1, 0])],
            ValueError,
            r"All elements in the list must be 1D numpy arrays\.",
        ),
        # A zero vector cannot be normalized.
        (
            [np.array([0, 0]), np.array([1, 0])],
            ValueError,
            r"Vectors must be nonzero to compute mutual coherence\.",
        ),
    ],
)
def test_mutual_coherence_invalid_inputs(vectors, exception, expected_msg):
    """Test that invalid inputs raise the exact ValueError or TypeError message."""
    with pytest.raises(exception, match=expected_msg):
        mutual_coherence(vectors)


def test_mutual_coherence_complex_vectors():
    """Complex vectors are not truncated; orthogonal complex vectors have coherence 0."""
    # <[1, 1j], [1j, 1]> = conj(1)*1j + conj(1j)*1 = 1j - 1j = 0, so these are orthogonal.
    result = mutual_coherence([np.array([1, 1j]), np.array([1j, 1])])
    assert np.isclose(result, 0.0, atol=1e-8)
