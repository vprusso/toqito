"""Test generate_random_linearly_independent_vectors."""

import numpy as np
import pytest

from toqito.matrix_props import is_linearly_independent
from toqito.rand.generate_random_linearly_independent_vectors import generate_random_linearly_independent_vectors


@pytest.mark.parametrize(
    "num_vecs,dim",
    [
        # Test two two-dimensional vectors.
        (2, 2),
        # Test four four-dimensional vectors.
        (4, 4),
        # Test ten ten-dimensional vectors.
        (10, 10),
        # Test two five-dimensional vectors.
        (2, 5),
        # Test two ten-dimensional vectors.
        (2, 10),
        # Test very high dimensiona;
        (999, 1000),
    ],
)
def test_generate_random_linearly_independent_vectors(num_vecs: int, dim: int):
    """Test for generate_random_linearly_independent_vectors function."""
    real_linear_indep = generate_random_linearly_independent_vectors(num_vecs, dim)
    complex_linear_indep = generate_random_linearly_independent_vectors(num_vecs, dim, is_real=False)

    # Verify the matrix has the correct dimensions.
    assert real_linear_indep.shape == (dim, num_vecs)
    assert complex_linear_indep.shape == (dim, num_vecs)

    # Verify the matrix is real.
    assert np.isreal(real_linear_indep).all()

    assert not np.isreal(complex_linear_indep).any()

    # verify the vectors are linearaly independent
    # by confirming the rank of the vector space
    # is equivalent to the number of vectors generated
    assert np.linalg.matrix_rank(real_linear_indep) == num_vecs
    assert is_linearly_independent(np.expand_dims(real_linear_indep.T, axis=2).tolist())
    assert np.linalg.matrix_rank(complex_linear_indep) == num_vecs
    assert is_linearly_independent(np.expand_dims(complex_linear_indep.T, axis=2).tolist())


@pytest.mark.parametrize(
    "num_vecs,dim",
    [
        # Test with a matrix of dimension 2.
        (3, 2),
        # Test with a matrix of higher dimension.
        (20, 15),
    ],
)
def test_generate_random_linearly_independent_vectors_failure(num_vecs: int, dim: int):
    """Test for generate_random_linearly_independent_vectors function; should fail."""
    with pytest.raises(ValueError) as excinfo:
        generate_random_linearly_independent_vectors(num_vecs, dim)
    assert excinfo.type is ValueError


@pytest.mark.parametrize(
    "num_vecs,dim,max_attempts",
    [
        # Test value error being raised
        (3, 2, 0),
        # Test what happens when max_attempts is null
        (3, 2, None),
    ],
)
def test_generate_random_linearly_independent_vectors_attempts(num_vecs: int, dim: int, max_attempts: int):
    """Test for generate_random_linearly_independent_vectors function; should fail."""
    with pytest.raises(ValueError) as excinfo:
        generate_random_linearly_independent_vectors(num_vecs, dim, max_attempts)
    assert excinfo.type is ValueError


def test_custom_max_attempts_failure(monkeypatch):
    """Test for exceed max_attempts."""
    # Monkey-patch matrix_rank to always return too-low rank
    calls = {"count": 0}

    def fake_rank(x):
        calls["count"] += 1
        # Always return one less than needed
        return x.shape[1] - 1

    monkeypatch.setattr(np.linalg, "matrix_rank", fake_rank)

    num, dim, max_attempts = 2, 3, 4
    with pytest.raises(ValueError) as excinfo:
        generate_random_linearly_independent_vectors(
            num_vectors=num, dim=dim, is_real=True, seed=123, max_attempts=max_attempts
        )
    assert f"after {max_attempts} attempts" in str(excinfo.value)
    # Ensure we actually tried max_attempts times
    assert calls["count"] == max_attempts


def test_default_max_attempts_used(monkeypatch):
    """Test tha max attempts defaults to num_vectors*dim."""
    # Ensure default max_attempts = num_vectors * dim
    calls = {"count": 0}

    def fake_rank(x):
        calls["count"] += 1
        return x.shape[1]  # always succeed on first try

    monkeypatch.setattr(np.linalg, "matrix_rank", fake_rank)
    num, dim = 2, 3
    # omit max_attempts to trigger default
    mat = generate_random_linearly_independent_vectors(
        num_vectors=num, dim=dim, is_real=True, seed=999, max_attempts=None
    )
    # default attempts = num*dim, but we succeed on first, so only one call
    assert calls["count"] == 1
    assert mat.shape == (dim, num)
