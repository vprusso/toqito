"""Test horodecki."""

import numpy as np
import pytest

from toqito.states import horodecki


def test_horodecki_state_3_3_default():
    """The 3-by-3 Horodecki state (no dimensions specified on input)."""
    expected_res = np.array(
        [
            [0.1000, 0, 0, 0, 0.1000, 0, 0, 0, 0.1000],
            [0, 0.1000, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0.1000, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0.1000, 0, 0, 0, 0, 0],
            [0.1000, 0, 0, 0, 0.1000, 0, 0, 0, 0.1000],
            [0, 0, 0, 0, 0, 0.1000, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0.1500, 0, 0.0866],
            [0, 0, 0, 0, 0, 0, 0, 0.1000, 0],
            [0.1000, 0, 0, 0, 0.1000, 0, 0.0866, 0, 0.1500],
        ]
    )

    res = horodecki(0.5)
    np.testing.assert_allclose(res, expected_res, atol=0.0001)


def test_horodecki_state_3_3():
    """The 3-by-3 Horodecki state."""
    expected_res = np.array(
        [
            [0.1000, 0, 0, 0, 0.1000, 0, 0, 0, 0.1000],
            [0, 0.1000, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0.1000, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0.1000, 0, 0, 0, 0, 0],
            [0.1000, 0, 0, 0, 0.1000, 0, 0, 0, 0.1000],
            [0, 0, 0, 0, 0, 0.1000, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0.1500, 0, 0.0866],
            [0, 0, 0, 0, 0, 0, 0, 0.1000, 0],
            [0.1000, 0, 0, 0, 0.1000, 0, 0.0866, 0, 0.1500],
        ]
    )

    res = horodecki(0.5, [3, 3])
    np.testing.assert_allclose(res, expected_res, atol=0.0001)


def test_horodecki_state_2_4():
    """The 2-by-4 Horodecki state."""
    expected_res = np.array(
        [
            [0.1111, 0, 0, 0, 0, 0.1111, 0, 0],
            [0, 0.1111, 0, 0, 0, 0, 0.1111, 0],
            [0, 0, 0.1111, 0, 0, 0, 0, 0.1111],
            [0, 0, 0, 0.1111, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0.1667, 0, 0.0962],
            [0.1111, 0, 0, 0, 0, 0.1111, 0, 0],
            [0, 0.1111, 0, 0, 0, 0, 0.1111, 0],
            [0, 0, 0.1111, 0, 0, 0.0962, 0, 0.1667],
        ]
    )

    res = horodecki(0.5, [2, 4])
    np.testing.assert_allclose(res, expected_res, atol=0.2)


@pytest.mark.parametrize(
    "a_param, dim",
    [
        # Invalid a_param (negative)."""
        (-5, None),
        # Invalid a_param."""
        (5, None),
        # Tests for invalid dimension inputs.
        (0.5, [3, 4]),
    ],
)
def test_horodecki_invalid(a_param, dim):
    """Tests for invalid a_param and dimension inputs."""
    with np.testing.assert_raises(ValueError):
        horodecki(a_param, dim)
