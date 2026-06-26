"""Test isotropic."""

import numpy as np
import pytest

from toqito.states import isotropic

iso_dim_3_alpha_half = np.array(
    [
        [0.22222222, 0, 0, 0, 0.16666667, 0, 0, 0, 0.16666667],
        [0, 0.05555556, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0.05555556, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0.05555556, 0, 0, 0, 0, 0],
        [0.16666667, 0, 0, 0, 0.22222222, 0, 0, 0, 0.16666667],
        [0, 0, 0, 0, 0, 0.05555556, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0.05555556, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0.05555556, 0],
        [0.16666667, 0, 0, 0, 0.16666667, 0, 0, 0, 0.22222222],
    ]
)


def test_isotropic_qutrit():
    """Generate a qutrit isotropic state with `alpha` = 1/2."""
    res = isotropic(3, 1 / 2)
    np.testing.assert_allclose(res, iso_dim_3_alpha_half)


def test_isotropic_alpha_out_of_range():
    """An alpha outside [-1/(d^2-1), 1] does not yield a valid isotropic state."""
    with pytest.raises(ValueError, match="must be in the interval"):
        isotropic(3, 2.0)
