"""Test measure."""

import numpy as np

from toqito.measurement_ops import measure
from toqito.matrices import standard_basis


def test_measure_state():
    """Test measure on quantum state."""
    e_0, e_1 = standard_basis(2)
    psi = 1 / np.sqrt(3) * e_0 + np.sqrt(2 / 3) * e_1
    rho = psi @ psi.conj().T

    proj_0 = e_0 @ e_0.conj().T
    proj_1 = e_1 @ e_1.conj().T
    np.testing.assert_equal(np.isclose(measure(proj_0, rho), 1 / 3), True)
    np.testing.assert_equal(np.isclose(measure(proj_1, rho), 2 / 3), True)
