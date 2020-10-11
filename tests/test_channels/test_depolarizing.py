"""Test depolarizing."""
import numpy as np

from toqito.channels import depolarizing
from toqito.channel_ops import apply_channel


def test_depolarizing_complete_depolarizing():
    """Maps every density matrix to the maximally-mixed state."""
    test_input_mat = np.array(
        [[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]]
    )

    expected_res = (
        1 / 4 * np.array([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]])
    )

    res = apply_channel(test_input_mat, depolarizing(4))

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_depolarizing_partially_depolarizing():
    """The partially depolarizing channel for `p = 0.5`."""
    test_input_mat = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])

    expected_res = np.array(
        [
            [17.125, 0.25, 0.375, 0.5],
            [0.625, 17.75, 0.875, 1],
            [1.125, 1.25, 18.375, 1.5],
            [1.625, 1.75, 1.875, 19],
        ]
    )

    res = apply_channel(test_input_mat, depolarizing(4, 0.5))

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
