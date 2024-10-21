"""Test singlet."""

import numpy as np

from toqito.states import bell, singlet


def test_gen_bell_dim_2():
    """Generalized singlet state for dim = 2."""
    dim = 2

    expected_res = bell(3) @ bell(3).conj().T

    res = singlet(dim)
    np.testing.assert_allclose(res, expected_res)
