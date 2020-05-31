"""Test isotropic."""
import numpy as np

from toqito.states import isotropic


def test_isotropic_qutrit():
    """Generate a qutrit isotropic state with `alpha` = 1/2."""
    res = isotropic(3, 1 / 2)

    np.testing.assert_equal(np.isclose(res[0, 0], 2 / 9), True)
    np.testing.assert_equal(np.isclose(res[4, 4], 2 / 9), True)
    np.testing.assert_equal(np.isclose(res[8, 8], 2 / 9), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
