"""Test negativity."""
import numpy as np

from toqito.state_props import negativity


def test_negativity_rho():
    """Test for negativity on rho."""
    test_input_mat = np.array(
        [[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]]
    )
    np.testing.assert_equal(np.isclose(negativity(test_input_mat), 1 / 2), True)


def test_negativity_rho_dim_int():
    """Test for negativity on rho."""
    test_input_mat = np.array(
        [[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]]
    )
    np.testing.assert_equal(np.isclose(negativity(test_input_mat, 2), 1 / 2), True)


def test_negativity_invalid_rho_dim_int():
    """Invalid dim parameters."""
    with np.testing.assert_raises(ValueError):
        test_input_mat = np.array(
            [[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]]
        )
        negativity(test_input_mat, 5)


def test_negativity_invalid_rho_dim_vec():
    """Invalid dim parameters."""
    with np.testing.assert_raises(ValueError):
        test_input_mat = np.array(
            [[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]]
        )
        negativity(test_input_mat, [2, 5])


if __name__ == "__main__":
    np.testing.run_module_suite()
