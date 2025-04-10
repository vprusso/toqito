"""Test random_povm."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from toqito.rand import random_povm


@pytest.mark.parametrize("dim", range(2, 8))
@pytest.mark.parametrize("num_inputs", range(2, 8))
@pytest.mark.parametrize("num_outputs", range(2, 8))
def test_random_povm_dimensions(dim, num_inputs, num_outputs):
    """Verify that the output has the correct shape as specified by the input parameters."""
    povms = random_povm(dim=dim, num_inputs=num_inputs, num_outputs=num_outputs)
    np.testing.assert_equal(povms.shape, (dim, dim, num_inputs, num_outputs))


@pytest.mark.parametrize("dim", range(2, 8))
@pytest.mark.parametrize("num_inputs", range(2, 8))
@pytest.mark.parametrize("num_outputs", range(2, 8))
def test_random_povm_validity(dim, num_inputs, num_outputs):
    """Each set of POVMs for a given input sums up to the identity matrix."""
    povms = random_povm(dim=dim, num_inputs=num_inputs, num_outputs=num_outputs)
    for i in range(num_inputs):
        sum_povms = sum(povms[:, :, i, j] for j in range(num_outputs))
        assert np.allclose(sum_povms, np.identity(dim))


@pytest.mark.parametrize(
    "dim, num_inputs, num_outputs, expected",
    [
        (
            2,
            2,
            2,
            np.array(
                [
                    [
                        [[0.68105648 + 0.0j, 0.31894352 + 0.0j], [0.46623871 + 0.0j, 0.53376129 + 0.0j]],
                        [[0.01373155 + 0.0j, -0.01373155 + 0.0j], [0.42523981 + 0.0j, -0.42523981 + 0.0j]],
                    ],
                    [
                        [[0.01373155 + 0.0j, -0.01373155 + 0.0j], [0.42523981 + 0.0j, -0.42523981 + 0.0j]],
                        [[0.04748388 + 0.0j, 0.95251612 + 0.0j], [0.47081969 + 0.0j, 0.52918031 + 0.0j]],
                    ],
                ]
            ),
        )
    ],
)
def test_seed(dim, num_inputs, num_outputs, expected):
    """Test that the function returns the expected output when seeded."""
    povms = random_povm(dim=dim, num_inputs=num_inputs, num_outputs=num_outputs, seed=123)
    assert_allclose(povms, expected, rtol=1e-05)
