"""Test log_negativity."""
import numpy as np

from toqito.state_props import log_negativity


def test_log_negativity_rho():
    """Test for log_negativity on rho."""
    test_input_mat = np.array(
        [[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]]
    )
    np.testing.assert_equal(np.isclose(log_negativity(test_input_mat), 1), True)


def test_log_negativity_rho_dim_int():
    """Test for log_negativity on rho."""
    test_input_mat = np.array(
        [[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]]
    )
    np.testing.assert_equal(np.isclose(log_negativity(test_input_mat, 2), 1), True)


def test_log_negativity_invalid_rho_dim_int():
    """Invalid dim parameters."""
    with np.testing.assert_raises(ValueError):
        test_input_mat = np.array(
            [[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]]
        )
        log_negativity(test_input_mat, 5)


def test_log_negativity_invalid_rho_dim_vec():
    """Invalid dim parameters."""
    with np.testing.assert_raises(ValueError):
        test_input_mat = np.array(
            [[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]]
        )
        log_negativity(test_input_mat, [2, 5])


if __name__ == "__main__":
    np.testing.run_module_suite()
