"""Test horodecki."""

import numpy as np
import pytest

from toqito.states import horodecki

EXPECTED_3_BY_3 = np.array(
    [
        [0.1, 0, 0, 0, 0.1, 0, 0, 0, 0.1],
        [0, 0.1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0.1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0.1, 0, 0, 0, 0, 0],
        [0.1, 0, 0, 0, 0.1, 0, 0, 0, 0.1],
        [0, 0, 0, 0, 0, 0.1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0.15, 0, 0.08660254037844387],
        [0, 0, 0, 0, 0, 0, 0, 0.1, 0],
        [0.1, 0, 0, 0, 0.1, 0, 0.08660254037844387, 0, 0.15],
    ]
)

EXPECTED_2_BY_4 = np.array(
    [
        [0.1111111111111111, 0, 0, 0, 0, 0.1111111111111111, 0, 0],
        [0, 0.1111111111111111, 0, 0, 0, 0, 0.1111111111111111, 0],
        [0, 0, 0.1111111111111111, 0, 0, 0, 0, 0.1111111111111111],
        [0, 0, 0, 0.1111111111111111, 0, 0, 0, 0],
        [0, 0, 0, 0, 0.16666666666666666, 0, 0, 0.09622504486493763],
        [0.1111111111111111, 0, 0, 0, 0, 0.1111111111111111, 0, 0],
        [0, 0.1111111111111111, 0, 0, 0, 0, 0.1111111111111111, 0],
        [0, 0, 0.1111111111111111, 0, 0.09622504486493763, 0, 0, 0.16666666666666666],
    ]
)


@pytest.mark.parametrize(
    ("dim", "expected_res"),
    [
        pytest.param(None, EXPECTED_3_BY_3, id="default-3-by-3"),
        pytest.param([3, 3], EXPECTED_3_BY_3, id="explicit-3-by-3"),
        pytest.param([2, 4], EXPECTED_2_BY_4, id="2-by-4"),
    ],
)
def test_horodecki_state_dimensions(dim, expected_res):
    """The Horodecki state for each supported dimension."""
    res = horodecki(0.5, dim)
    np.testing.assert_allclose(res, expected_res, rtol=0, atol=1e-12)


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
