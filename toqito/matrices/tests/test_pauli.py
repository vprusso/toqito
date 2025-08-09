"""Test pauli."""

import numpy as np
import pytest
from scipy.sparse import issparse

from toqito.matrices import pauli


@pytest.mark.parametrize(
    "pauli_ind_input,expected",
    [
        ("i", np.array([[1, 0], [0, 1]])),
        ("I", np.array([[1, 0], [0, 1]])),
        (0, np.array([[1, 0], [0, 1]])),
        ("X", np.array([[0, 1], [1, 0]])),
        ("x", np.array([[0, 1], [1, 0]])),
        (1, np.array([[0, 1], [1, 0]])),
        ("Y", np.array([[0, -1j], [1j, 0]])),
        ("y", np.array([[0, -1j], [1j, 0]])),
        (2, np.array([[0, -1j], [1j, 0]])),
        ("Z", np.array([[1, 0], [0, -1]])),
        ("z", np.array([[1, 0], [0, -1]])),
        (3, np.array([[1, 0], [0, -1]])),
    ],
)
def test_pauli_single(pauli_ind_input, expected):
    """Test single Pauli operators with string and integer inputs."""
    result = pauli(pauli_ind_input)
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("pauli_ind_input", ["I", 0, "X", 1, "Y", 2, "Z", 3, "i", "x", "y", "z"])
def test_pauli_sparse(pauli_ind_input):
    """Test that sparse flag produces sparse matrices."""
    result = pauli(pauli_ind_input, True)
    assert issparse(result)


@pytest.mark.parametrize(
    "pauli_list,expected",
    [
        ([1, 1], np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])),
        (["X", "X"], np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])),
    ],
)
def test_pauli_list(pauli_list, expected):
    """Test tensor products of Pauli operators."""
    result = pauli(pauli_list)
    np.testing.assert_allclose(result, expected)


def test_invalid_ind():
    """Invalid input for pauli operator index."""
    with pytest.raises(ValueError, match=r"Invalid Pauli operator index provided"):
        pauli(4)
