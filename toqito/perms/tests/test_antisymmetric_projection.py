"""Test antisymmetric_projection."""

import numpy as np
import pytest

from toqito.perms import antisymmetric_projection

# Create a zero vector of length 27
anti_proj_3_3_partial = np.zeros((27, 1))

# Set specific indices to -0.40824829 and 0.40824829
anti_proj_3_3_partial[5] = -0.40824829
anti_proj_3_3_partial[7] = 0.40824829
anti_proj_3_3_partial[11] = 0.40824829
anti_proj_3_3_partial[15] = -0.40824829
anti_proj_3_3_partial[19] = -0.40824829
anti_proj_3_3_partial[21] = 0.40824829


@pytest.mark.parametrize(
    "dim, p_param, partial, expected_result",
    [
        # Dimension is 2 and p is equal to 1.
        (2, 1, False, np.array([[1, 0], [0, 1]])),
        # The `p` value is greater than the dimension `d`.
        (2, 3, False, np.zeros((8, 8))),
        # The dimension is 2.
        (2, 2, False, np.array([[0, 0, 0, 0], [0, 0.5, -0.5, 0], [0, -0.5, 0.5, 0], [0, 0, 0, 0]])),
        # The `dim` is 3, the `p` is 3, and `partial` is True.
        (3, 3, True, anti_proj_3_3_partial),
    ],
)
def test_antisymmetric_projection(dim, p_param, partial, expected_result):
    """Test function works as expected for a valid input."""
    proj = antisymmetric_projection(dim=dim, p_param=p_param, partial=partial)
    assert abs(proj - expected_result).all() <= 1e-3
