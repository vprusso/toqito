"""Test random_ginibre."""

import numpy as np
import pytest

from toqito.rand import random_ginibre


@pytest.mark.parametrize("dim_n", range(0, 8))
@pytest.mark.parametrize("dim_m", range(0, 8))
def test_random_ginibre_dims(dim_n, dim_m):
    """Generate random Ginibre matrix and check proper dimensions."""
    gin_mat = random_ginibre(dim_n, dim_m)
    np.testing.assert_equal(gin_mat.shape, (dim_n, dim_m))


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
