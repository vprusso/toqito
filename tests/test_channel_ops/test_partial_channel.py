"""Tests for partial_channel."""
import numpy as np

from toqito.channel_ops import partial_channel
from toqito.channels import depolarizing


def test_partial_channel_depolarizing_first_system():
    """
    Perform the partial map using the depolarizing channel as the Choi
    matrix on first system.
    """
    rho = np.array(
        [
            [
                0.3500,
                -0.1220 - 0.0219 * 1j,
                -0.1671 - 0.0030 * 1j,
                -0.1170 - 0.0694 * 1j,
            ],
            [
                -0.0233 + 0.0219 * 1j,
                0.1228,
                -0.2775 + 0.0492 * 1j,
                -0.2613 + 0.0529 * 1j,
            ],
            [
                -0.2671 + 0.0030 * 1j,
                -0.2775 - 0.0492 * 1j,
                0.1361,
                0.0202 + 0.0062 * 1j,
            ],
            [
                -0.2170 + 0.0694 * 1j,
                -0.2613 - 0.0529 * 1j,
                0.2602 - 0.0062 * 1j,
                0.2530,
            ],
        ]
    )
    res = partial_channel(rho, depolarizing(2))

    expected_res = np.array(
        [
            [0.2364 + 0.0j, 0.0 + 0.0j, -0.2142 + 0.02495j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.2364 + 0.0j, 0.0 + 0.0j, -0.2142 + 0.02495j],
            [-0.2642 - 0.02495j, 0.0 + 0.0j, 0.19455 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, -0.2642 - 0.02495j, 0.0 + 0.0j, 0.19455 + 0.0j],
        ]
    )

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_partial_channel_depolarizing_second_system():
    """
    Perform the partial map using the depolarizing channel as the Choi
    matrix on second system.
    """
    rho = np.array(
        [
            [
                0.3101,
                -0.0220 - 0.0219 * 1j,
                -0.0671 - 0.0030 * 1j,
                -0.0170 - 0.0694 * 1j,
            ],
            [
                -0.0220 + 0.0219 * 1j,
                0.1008,
                -0.0775 + 0.0492 * 1j,
                -0.0613 + 0.0529 * 1j,
            ],
            [
                -0.0671 + 0.0030 * 1j,
                -0.0775 - 0.0492 * 1j,
                0.1361,
                0.0602 + 0.0062 * 1j,
            ],
            [
                -0.0170 + 0.0694 * 1j,
                -0.0613 - 0.0529 * 1j,
                0.0602 - 0.0062 * 1j,
                0.4530,
            ],
        ]
    )
    res = partial_channel(rho, depolarizing(2), 1)

    expected_res = np.array(
        [
            [0.2231 + 0.0j, 0.0191 - 0.00785j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0191 + 0.00785j, 0.2769 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.2231 + 0.0j, 0.0191 - 0.00785j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0191 + 0.00785j, 0.2769 + 0.0j],
        ]
    )

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_partial_channel_dim_list():
    """
    Perform the partial map using the depolarizing channel as the Choi
    matrix on first system when the dimension is specified as list.
    """
    rho = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    res = partial_channel(rho, depolarizing(2), 2, [2, 2])

    expected_res = np.array(
        [
            [3.5, 0.0, 5.5, 0.0],
            [0.0, 3.5, 0.0, 5.5],
            [11.5, 0.0, 13.5, 0.0],
            [0.0, 11.5, 0.0, 13.5],
        ]
    )

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_partial_channel_non_square_matrix():
    """Matrix must be square."""
    with np.testing.assert_raises(ValueError):
        rho = np.array([[1, 1, 1, 1], [5, 6, 7, 8], [3, 3, 3, 3]])
        partial_channel(rho, depolarizing(3))


def test_partial_channel_non_square_matrix_2():
    """Matrix must be square with sys arg."""
    with np.testing.assert_raises(ValueError):
        rho = np.array([[1, 2, 3, 4], [2, 2, 2, 2], [12, 11, 10, 9]])
        partial_channel(rho, depolarizing(3), 2)


def test_partial_channel_invalid_dim():
    """Invalid dimension for partial map."""
    with np.testing.assert_raises(ValueError):
        rho = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [12, 11, 10, 9]])
        partial_channel(rho, depolarizing(3), 1, [2, 2])


def test_partial_channel_invalid_map():
    """Invalid map argument for partial map."""
    with np.testing.assert_raises(ValueError):
        rho = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [12, 11, 10, 9]])
        partial_channel(rho, 5)


if __name__ == "__main__":
    np.testing.run_module_suite()
