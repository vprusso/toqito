"""Test BB84."""

import numpy as np

from toqito.states import bb84


def test_bb84():
    """Test function generates the correct BB84 states."""
    e_0, e_1 = np.array([[1], [0]]), np.array([[0], [1]])
    e_p, e_m = (e_0 + e_1) / np.sqrt(2), (e_0 - e_1) / np.sqrt(2)

    states = bb84()

    np.testing.assert_array_equal(states[0][0], e_0)
    np.testing.assert_array_equal(states[0][1], e_1)
    np.testing.assert_array_equal(states[1][0], e_p)
    np.testing.assert_array_equal(states[1][1], e_m)
