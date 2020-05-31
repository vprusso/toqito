"""Test is_psd."""
import numpy as np

from toqito.matrix_props import is_psd


def test_is_psd():
    """Test that positive semidefinite matrix returns True."""
    mat = np.array([[1, -1], [-1, 1]])
    np.testing.assert_equal(is_psd(mat), True)


def test_is_not_psd():
    """Test that non-positive semidefinite matrix returns False."""
    mat = np.array([[-1, -1], [-1, -1]])
    np.testing.assert_equal(is_psd(mat), False)


if __name__ == "__main__":
    np.testing.run_module_suite()
