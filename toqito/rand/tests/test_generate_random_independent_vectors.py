"""Unit tests for generate_random_independent_vectors."""

import numpy as np
import pytest

from toqito.matrix_props import is_linearly_independent
from toqito.rand.generate_random_independent_vectors import (
    generate_random_independent_vectors,
)


@pytest.mark.parametrize("is_real", [True, False])
def test_generate_random_independent_vectors_valid(is_real):
    """Generated vectors should be linearly independent."""
    vecs = generate_random_independent_vectors(
        num_vectors=3,
        dim=5,
        is_real=is_real,
        seed=123,
    )

    assert vecs.shape == (5, 3)

    vectors = [vecs[:, i] for i in range(vecs.shape[1])]
    assert is_linearly_independent(vectors)


def test_generate_random_independent_vectors_reproducible():
    """Generator should be reproducible when a seed is provided."""
    vecs1 = generate_random_independent_vectors(2, 4, seed=42)
    vecs2 = generate_random_independent_vectors(2, 4, seed=42)

    assert np.allclose(vecs1, vecs2)


def test_generate_random_independent_vectors_invalid():
    """Requesting too many vectors should raise a ValueError."""
    with pytest.raises(ValueError):
        generate_random_independent_vectors(num_vectors=5, dim=3)

def test_generate_random_independent_vectors_failure(monkeypatch):
    """Raises RuntimeError if independent vectors cannot be generated."""

    def always_dependent(*args, **kwargs):
        return np.zeros((3, 2))

    monkeypatch.setattr(
        "toqito.rand.generate_random_independent_vectors.is_linearly_independent",
        lambda _: False,
    )

    with pytest.raises(RuntimeError):
        generate_random_independent_vectors(num_vectors=2, dim=3, seed=0)
