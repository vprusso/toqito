"""Test perturb vectors."""

import numpy as np
import pytest

from toqito.matrix_ops import perturb_vectors


@pytest.mark.parametrize(
    "vectors, eps, expected_length",
    [
        # Test with three vectors along the axes
        ([np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])], 0.1, 3),
        # Test with one vector and eps=0
        ([np.array([1, 1, 0])], 0.0, 1),
        # Test with two vectors and a different perturbation value
        ([np.array([1, 0, 0]), np.array([0, 1, 0])], 0.2, 2),
    ],
)
def test_output_size(vectors, eps, expected_length):
    """Test that the function returns the same number of vectors as input."""
    perturbed_vectors = perturb_vectors(vectors, eps)
    assert len(perturbed_vectors) == expected_length


@pytest.mark.parametrize(
    "vectors, eps",
    [
        ([np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])], 0.1),
        ([np.array([1, 1, 1])], 0.1),
    ],
)
def test_normalization(vectors, eps):
    """Test that each perturbed vector is normalized."""
    perturbed_vectors = perturb_vectors(vectors, eps)
    for pv in perturbed_vectors:
        norm = np.linalg.norm(pv)
        assert np.isclose(norm, 1.0, atol=1e-5)


@pytest.mark.parametrize(
    "vectors, eps",
    [
        ([np.array([1, 0, 0]), np.array([0, 1, 0])], 0.1),
        ([np.array([1, 1, 1])], 0.2),
    ],
)
def test_perturbation_effect(vectors, eps):
    """Test that the perturbed vectors are different from the original vectors."""
    perturbed_vectors = perturb_vectors(vectors, eps)
    for i in range(len(vectors)):
        assert not np.array_equal(vectors[i], perturbed_vectors[i])


@pytest.mark.parametrize(
    "vectors",
    [
        ([np.array([1, 0, 0]), np.array([0, 1, 0])]),
        ([np.array([1, 1, 1])]),
    ],
)
def test_zero_perturbation(vectors):
    """Test that if eps = 0, the vectors remain the same."""
    perturbed_vectors = perturb_vectors(vectors, eps=0.0)
    for i in range(len(vectors)):
        assert np.allclose(vectors[i], perturbed_vectors[i]), f"Vector {i} does not match."
