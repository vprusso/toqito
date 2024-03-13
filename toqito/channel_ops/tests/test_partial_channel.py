"""Tests for partial_channel."""

import numpy as np
import pytest

from toqito.channel_ops import partial_channel
from toqito.channels import depolarizing
from toqito.matrices import pauli

rho_first_system = np.array(
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

expected_res_first_system = np.array(
    [
        [0.2364 + 0.0j, 0.0 + 0.0j, -0.2142 + 0.02495j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.2364 + 0.0j, 0.0 + 0.0j, -0.2142 + 0.02495j],
        [-0.2642 - 0.02495j, 0.0 + 0.0j, 0.19455 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, -0.2642 - 0.02495j, 0.0 + 0.0j, 0.19455 + 0.0j],
    ]
)


rho_second_system = np.array(
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

expected_res_second_system = np.array(
    [
        [0.2231 + 0.0j, 0.0191 - 0.00785j, 0.0 + 0.0j, 0.0 + 0.0j],
        [0.0191 + 0.00785j, 0.2769 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j, 0.2231 + 0.0j, 0.0191 - 0.00785j],
        [0.0 + 0.0j, 0.0 + 0.0j, 0.0191 + 0.00785j, 0.2769 + 0.0j],
    ]
)

rho_dim_list = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])

expected_res_dim_list = np.array(
    [
        [3.5, 0.0, 5.5, 0.0],
        [0.0, 3.5, 0.0, 5.5],
        [11.5, 0.0, 13.5, 0.0],
        [0.0, 11.5, 0.0, 13.5],
    ]
)


@pytest.mark.parametrize(
    "test_input, expected, sys_arg, dim_arg",
    [
        # Perform the partial map using the depolarizing channel as the Choi matrix on first system
        (rho_first_system, expected_res_first_system, None, None),
        # Perform the partial map using the depolarizing channel as the Choi matrix on second system
        (rho_second_system, expected_res_second_system, 1, None),
        # Test uses the depolarizing channel as the Choi matrix on first system when the dimension is specified as list.
        (rho_dim_list, expected_res_dim_list, 2, [2, 2]),
    ],
)
def test_partial_channel(test_input, expected, sys_arg, dim_arg):
    """Test function works as expected for valid inputs."""
    if sys_arg is None and dim_arg is None:
        calculated = partial_channel(test_input, depolarizing(2))
    elif sys_arg is not None and dim_arg is None:
        calculated = partial_channel(test_input, depolarizing(2), sys_arg)
    elif sys_arg is not None and dim_arg is not None:
        calculated = partial_channel(test_input, depolarizing(2), sys_arg, dim_arg)

    assert np.isclose(calculated, expected).all()


@pytest.mark.parametrize("nested", [1, 2, 3])
def test_partial_channel_cpt_kraus(nested):
    """Perform the partial map using the Kraus representation of the depolarizing channel."""
    rho = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    kraus = [0.5 * pauli(ind) for ind in range(4)]
    if nested == 2:
        kraus = [kraus]
    elif nested == 3:
        kraus = [[mat] for mat in kraus]

    res = partial_channel(rho, kraus)

    expected_res = np.array(
        [
            [3.5, 0.0, 5.5, 0.0],
            [0.0, 3.5, 0.0, 5.5],
            [11.5, 0.0, 13.5, 0.0],
            [0.0, 11.5, 0.0, 13.5],
        ]
    )

    assert np.isclose(expected_res, res).all()


@pytest.mark.parametrize(
    "test_input, map_arg, sys_arg, dim_arg",
    [
        # Matrix must be square
        (np.array([[1, 1, 1, 1], [5, 6, 7, 8], [3, 3, 3, 3]]), depolarizing(3), None, None),
        # Matrix must be square with sys arg
        (np.array([[1, 2, 3, 4], [2, 2, 2, 2], [12, 11, 10, 9]]), depolarizing(3), 2, None),
        # Invalid dimension for partial map
        (np.array([[1, 2, 3, 4], [5, 6, 7, 8], [12, 11, 10, 9]]), depolarizing(3), 1, [2, 2]),
        # Invalid map argument for partial map
        (np.array([[1, 2, 3, 4], [5, 6, 7, 8], [12, 11, 10, 9]]), 5, 2, [2, 2]),
    ],
)
def test_partial_channel_error(test_input, map_arg, sys_arg, dim_arg):
    """Test function raises error as expected for invalid inputs."""
    if sys_arg is not None:
        with pytest.raises(ValueError):
            partial_channel(test_input, map_arg, sys_arg, dim_arg)

    with pytest.raises(ValueError):
        partial_channel(test_input, map_arg)
