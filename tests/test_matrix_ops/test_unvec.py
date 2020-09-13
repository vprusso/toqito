"""Test unvec."""
import numpy as np

from toqito.matrix_ops import unvec


def test_unvec():
    """Test standard unvec operation on a vector."""
    expected_res = np.array([[1, 2], [3, 4]])

    test_input_vec = np.array([1, 3, 2, 4])

    res = unvec(test_input_vec)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_unvec_custom_dim():
    """Test standard unvec operation on a vector with custom dimension."""
    expected_res = np.array([[1], [3], [2], [4]])

    test_input_vec = np.array([1, 3, 2, 4])

    res = unvec(test_input_vec, [1, 4])

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
