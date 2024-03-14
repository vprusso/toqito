"""Test isotropic."""

import numpy as np

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
