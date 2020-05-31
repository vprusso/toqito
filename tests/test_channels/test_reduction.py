"""Test reduction."""
import numpy as np

from toqito.channels import reduction


def test_reduction_map():
    """Test for the standard reduction map."""
    res = reduction(3)
    np.testing.assert_equal(res[4, 0], -1)
    np.testing.assert_equal(res[8, 0], -1)
    np.testing.assert_equal(res[1, 1], 1)
    np.testing.assert_equal(res[2, 2], 1)
    np.testing.assert_equal(res[3, 3], 1)
    np.testing.assert_equal(res[0, 4], -1)
    np.testing.assert_equal(res[8, 4], -1)
    np.testing.assert_equal(res[5, 5], 1)
    np.testing.assert_equal(res[6, 6], 1)
    np.testing.assert_equal(res[7, 7], 1)
    np.testing.assert_equal(res[0, 8], -1)
    np.testing.assert_equal(res[4, 8], -1)


if __name__ == "__main__":
    np.testing.run_module_suite()
