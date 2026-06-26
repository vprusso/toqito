"""Test random_state_vector."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from toqito.rand import random_state_vector
from toqito.state_props import is_pure


@pytest.mark.parametrize(
    "dim, is_real, k_param",
    [
        # Check that complex state vector from random state vector is pure.
        (2, False, 0),
        # Check that complex state vector with k_param > 0.
        (2, False, 1),
        # Check that complex state vector with k_param > 0 and dim list.
        ([2, 2], False, 1),
        # Check that complex state vector with k_param == 0 and dim list.
        ([2, 2], False, 0),
        # Check that real state vector with k_param > 0.
        (2, True, 1),
        # Check that real state vector from random state vector is pure.
        (2, True, 0),
    ],
)
def test_random_state_vector(dim, is_real, k_param):
    """Test function works as expected for a valid input."""
    # We expect the density matrix of any random state vector to be pure.
    vec = random_state_vector(dim=dim, is_real=is_real, k_param=k_param).reshape(-1, 1)
    mat = vec @ vec.conj().T
    assert is_pure(mat)


@pytest.mark.parametrize(
    "dim, is_real, k_param, expected",
    [
        (
            2,
            False,
            0,
            np.array([[-0.59005974 + 0.76831103j], [-0.2194029 + 0.11571532j]]),
        ),
        (
            2,
            False,
            1,
            np.array(
                [
                    [-0.2936362 + 0.77427174j],
                    [-0.29464507 - 0.15255447j],
                    [-0.04538643 + 0.41699573j],
                    [-0.1638817 - 0.03728129j],
                ]
            ),
        ),
        (
            [2, 2],
            False,
            1,
            np.array(
                [
                    [-0.2936362 + 0.77427174j],
                    [-0.29464507 - 0.15255447j],
                    [-0.04538643 + 0.41699573j],
                    [-0.1638817 - 0.03728129j],
                ]
            ),
        ),
        (
            2,
            True,
            1,
            np.array(
                [
                    [-0.92684881],
                    [-0.1395927],
                    [-0.34463175],
                    [-0.05190499],
                ]
            ),
        ),
        (2, True, 0, np.array([[-0.93730189], [-0.34851853]])),
    ],
)
def test_seed(dim, is_real, k_param, expected):
    """Test that the function returns the expected output when seeded."""
    vec = random_state_vector(dim, is_real=is_real, k_param=k_param, seed=123)
    assert_allclose(vec, expected)


@pytest.mark.parametrize("is_real", [False, True])
def test_random_state_vector_mean_outer_is_maximally_mixed(is_real):
    """Averaging |psi><psi| over Haar-uniform pure states approaches I/d.

    With the previous uniform (rather than Gaussian) sampling the vectors were confined to the positive orthant, so
    the average outer product had large positive off-diagonal entries and did not converge to I/d.
    """
    dim, n_samples = 2, 4000
    acc = np.zeros((dim, dim), dtype=complex)
    for s in range(n_samples):
        vec = random_state_vector(dim, is_real=is_real, seed=s)
        acc += vec @ vec.conj().T
    np.testing.assert_allclose(acc / n_samples, np.eye(dim) / dim, atol=0.02)


def test_random_state_vector_negative_k_param():
    """k_param must be non-negative."""
    with pytest.raises(ValueError, match="k_param must be non-negative"):
        random_state_vector(2, k_param=-1)


def test_random_state_vector_empty_dim_sequence():
    """Empty dimension sequences are not allowed."""
    with pytest.raises(ValueError, match="dim must not be empty"):
        random_state_vector([], k_param=0)


def test_random_state_vector_non_positive_dimension():
    """Dimension entries must be positive integers."""
    with pytest.raises(ValueError, match="dim entries must be positive integers"):
        random_state_vector([2, 0], k_param=0)


def test_random_state_vector_invalid_bipartite_shape_for_k_param():
    """When k_param > 0 the dimension sequence must describe a bipartite system."""
    with pytest.raises(ValueError, match="When k_param > 0"):
        random_state_vector([2, 2, 2], k_param=1)
