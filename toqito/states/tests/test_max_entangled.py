"""Test max_entangled."""

import numpy as np
import pytest

from toqito.states import max_entangled

e_0, e_1 = np.array([[1], [0]]), np.array([[0], [1]])


@pytest.mark.parametrize(
    "dim, is_sparse, is_normalized, expected_res",
    [
        # Generate maximally entangled state: `1/sqrt(2) * (|00> + |11>)`."""
        (2, False, True, 1 / np.sqrt(2) * (np.kron(e_0, e_0) + np.kron(e_1, e_1))),
        # Generate maximally entangled state: `|00> + |11>`.
        (2, False, False, 1 * (np.kron(e_0, e_0) + np.kron(e_1, e_1))),
    ],
)
def test_max_entangled(dim, is_sparse, is_normalized, expected_res):
    """Test function works as expected for a valid input."""
    res = max_entangled(dim=dim, is_sparse=is_sparse, is_normalized=is_normalized)
    np.testing.assert_allclose(res, expected_res)
