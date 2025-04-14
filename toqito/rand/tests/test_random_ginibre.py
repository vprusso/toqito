"""Test random_ginibre."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from toqito.rand import random_ginibre


@pytest.mark.parametrize("dim_n", range(0, 8))
@pytest.mark.parametrize("dim_m", range(0, 8))
def test_random_ginibre_dims(dim_n, dim_m):
    """Generate random Ginibre matrix and check proper dimensions."""
    gin_mat = random_ginibre(dim_n, dim_m)
    np.testing.assert_equal(gin_mat.shape, (dim_n, dim_m))


@pytest.mark.parametrize(
    "dim_n, dim_m, expected",
    [
        (
            2,
            3,
            np.array(
                [
                    [-0.69941441 - 0.45004776j, -0.26006444 + 0.38321809j, 0.91070069 - 0.22386679j],
                    [0.13716063 - 0.22796353j, 0.65070151 + 0.06870767j, 0.408074 - 1.07899574j],
                ]
            ),
        ),
        (
            2,
            2,
            np.array(
                [
                    [-0.69941441 + 0.65070151j, -0.26006444 + 0.408074j],
                    [0.91070069 - 0.45004776j, 0.13716063 + 0.38321809j],
                ]
            ),
        ),
    ],
)
def test_seed(dim_n, dim_m, expected):
    """Test that the function returns the expected output when seeded."""
    gin_mat = random_ginibre(dim_n, dim_m, seed=123)
    assert_allclose(gin_mat, expected)


@pytest.mark.parametrize(
    "dim_n, dim_m",
    [
        # Negative dim_n.
        (-1, 3),
        # Negative dim_m.
        (3, -2),
        # Negative dim_n and dim_m.
        (-4, -2),
    ],
)
def test_random_ginibre_negative_dims(dim_n, dim_m):
    """Negative dimensions are not allowed."""
    with pytest.raises(ValueError, match="negative dimensions are not allowed"):
        random_ginibre(dim_n, dim_m)
