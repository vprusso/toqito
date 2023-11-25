"""Test brauer."""
import numpy as np
import pytest

from toqito.states import brauer

brauer_2_2 = np.array(
    [
        [1, 1, 1],
        [0, 0, 0],
        [0, 0, 0],
        [1, 0, 0],
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 0, 0],
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [1, 1, 1],
    ]
)


@pytest.mark.parametrize(
    "dim, p_val, expected_result",
    [
        # Generate Brauer states on 4 qubits.
        (2, 2, brauer_2_2),
    ],
)
def test_brauer(dim, p_val, expected_result):
    np.testing.assert_array_equal(brauer(dim, p_val), expected_result)
