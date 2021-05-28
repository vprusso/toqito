"""Test singlet."""
import numpy as np

from toqito.states import bell
from toqito.states import singlet


def test_gen_bell_dim_2():
    """Generalized singlet state for dim = 2."""
    dim = 2

    expected_res = bell(3) * bell(3).conj().T

    res = singlet(dim)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
