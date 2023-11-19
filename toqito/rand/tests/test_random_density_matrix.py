"""Test random_density_matrix."""
import numpy as np
import pytest

from toqito.matrix_props import is_density
from toqito.rand import random_density_matrix


@pytest.mark.parametrize("dim", range(2, 8))
@pytest.mark.parametrize("is_real", [True, False])
@pytest.mark.parametrize("k_param", range(2, 4))
@pytest.mark.parametrize("distance_metric", ["bures", "haar"])
def test_random_density(dim, is_real, k_param, distance_metric):
    if k_param == dim:
        mat = random_density_matrix(
            dim=dim, is_real=is_real, k_param=k_param, distance_metric=distance_metric
        )
        np.testing.assert_equal(is_density(mat), True)
