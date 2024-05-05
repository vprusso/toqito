"""Test gen_pauli_z."""

import numpy as np
import pytest

from toqito.matrices import gen_pauli_z


@pytest.mark.parametrize("dim", range(2, 1024, 64))
def test_shape(dim):
    """Ensure the shape of a generalized Pauli Z is what we expect."""
    assert gen_pauli_z(dim).shape == (dim, dim)


@pytest.mark.parametrize(
    "dim, want",
    [
        (1, np.array([[1.0]])),
        (
            3,
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, -0.5 + 0.8660254j, 0.0],
                    [0.0, 0.0, -0.5 - 0.8660254j],
                ]
            ),
        ),
        (
            9,
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.76604444 + 0.64278761j, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.17364818 + 0.98480775j, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, -0.5 + 0.8660254j, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, -0.93969262 + 0.34202014j, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, -0.93969262 - 0.34202014j, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5 - 0.8660254j, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.17364818 - 0.98480775j, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.76604444 - 0.64278761j],
                ]
            ),
        ),
    ],
)
def test_values(dim, want):
    """Ensure the values of a generalized Pauli Z are what we expect."""
    assert np.isclose(gen_pauli_z(dim), want).all()
