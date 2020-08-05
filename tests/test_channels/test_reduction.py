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


def test_reduction_map_dim_3_k_2():
    """Test for the reduction map with dimension 3 and parameter k = 2."""
    res = reduction(3, 2)
    expected_res = np.array(
        [
            [1, 0, 0, 0, -1, 0, 0, 0, -1],
            [0, 2, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 2, 0, 0, 0, 0, 0],
            [-1, 0, 0, 0, 1, 0, 0, 0, -1],
            [0, 0, 0, 0, 0, 2, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 2, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 2, 0],
            [-1, 0, 0, 0, -1, 0, 0, 0, 1],
        ]
    )

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
