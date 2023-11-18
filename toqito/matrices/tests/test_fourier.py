"""Test fourier."""
import numpy as np

from toqito.matrices import fourier


def test_fourier_dim_2():
    """Fourier matrix of dimension 2."""
    expected_res = np.array([[1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), -1 / np.sqrt(2)]])

    res = fourier(2)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
