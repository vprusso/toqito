"""Test gen_bell."""

import numpy as np
import pytest

from toqito.states import bell, gen_bell


@pytest.mark.parametrize(
    "k_1, k_2, dim, expected_result",
    [
        # Generalized Bell state for k_1 = k_2 = 0 and dim = 2.
        (0, 0, 2, bell(0) @ bell(0).conj().T),
        # Generalized Bell state for k_1 = 0, k_2 = 1 and dim = 2.
        (0, 1, 2, bell(1) @ bell(1).conj().T),
        # Generalized Bell state for k_1 = 1, k_2 = 0 and dim = 2.
        (1, 0, 2, bell(2) @ bell(2).conj().T),
        # Generalized Bell state for k_1 = 1, k_2 = 1 and dim = 2.
        (1, 1, 2, bell(3) @ bell(3).conj().T),
    ],
)
def test_gen_bell(k_1, k_2, dim, expected_result):
    """Test function works as expected for a valid input."""
    np.testing.assert_allclose(gen_bell(k_1, k_2, dim), expected_result)
